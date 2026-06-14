"""Draft session index — unified train + val lookups for the dashboard.

Two backends, one façade:

  - **Val** loads `data/test/draft_sessions.json` (small, ~5K drafts) fully
    into memory.
  - **Train** lazy-reads `data/train/draft_sessions.jsonl` via a binary index
    (`draft_sessions.idx.bin`) — sorted by `draft_id`, mmapped, bisected for
    O(log N) point lookups without parsing 1.5 GB of JSON at boot.

`draft_sessions` (module-level singleton) is the `UnifiedDraftSessionsService`.
Every caller (routers, services) gets train + val transparently; each
`DraftSession` carries a `split: "train" | "val"` field for UI distinction.

On-disk integrity: the train manifest holds a sha256 of the JSONL it was built
against. The service refuses to load on mismatch — protects against partial
bootstrap crashes serving torn data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mmap
import threading
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Literal

from backend.lib.config import TEST_DIR, TRAIN_DIR


log = logging.getLogger(__name__)


Split = Literal["train", "val"]


# Train idx record layout — must stay in sync with scripts/process_data.js
# writeTrainDraftSidecars: 12-byte ASCII draft_id, 8-byte u64 offset (LE),
# 4-byte u32 length (LE).
DRAFT_ID_BYTES = 12
IDX_RECORD_BYTES = DRAFT_ID_BYTES + 8 + 4


@dataclass(slots=True)
class DraftSession:
    draft_id: str
    cube_uuid: str | None
    owner: str | None
    picks: list[int]
    synthetic: bool
    suspect: bool
    split: Split


# ============================================================================
# Val: tiny in-memory index, fully loaded at first access.
# ============================================================================


class ValDraftSessionsService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            _ = self.sessions
            _ = self._by_uuid
            _ = self._picks_by_uuid
            self._loaded = True

    @cached_property
    def sessions(self) -> dict[str, DraftSession]:
        path = TEST_DIR / "draft_sessions.json"
        if not path.is_file():
            return {}
        raw = json.loads(path.read_text())
        return {
            draft_id: DraftSession(
                draft_id=draft_id,
                cube_uuid=body.get("cube_uuid"),
                owner=body.get("owner"),
                picks=body["picks"],
                synthetic=bool(body.get("synthetic", False)),
                suspect=bool(body.get("suspect", False)),
                split="val",
            )
            for draft_id, body in raw.get("sessions", {}).items()
        }

    @cached_property
    def _by_uuid(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for s in self.sessions.values():
            if s.cube_uuid:
                out.setdefault(s.cube_uuid, []).append(s.draft_id)
        return out

    @cached_property
    def _picks_by_uuid(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for s in self.sessions.values():
            if s.cube_uuid:
                out.setdefault(s.cube_uuid, []).extend(s.picks)
        return out

    def get(self, draft_id: str) -> DraftSession | None:
        self.ensure_loaded()
        return self.sessions.get(draft_id)

    def list_by_cube(self, cube_uuid: str) -> list[DraftSession]:
        self.ensure_loaded()
        ids = self._by_uuid.get(cube_uuid, [])
        return [self.sessions[i] for i in ids]

    def pick_indices_by_cube(self, cube_uuid: str) -> list[int]:
        self.ensure_loaded()
        return self._picks_by_uuid.get(cube_uuid, [])

    def list_all(self, include_suspect: bool = False, limit: int = 100, offset: int = 0) -> list[DraftSession]:
        self.ensure_loaded()
        out = [s for s in self.sessions.values() if include_suspect or not s.suspect]
        out.sort(key=lambda s: s.picks[0] if s.picks else 0)
        return out[offset : offset + limit]


# ============================================================================
# Train: lazy mmap + bisect. No body data loaded at boot beyond the cube index.
# ============================================================================


class TrainDraftSessionsService:
    """Boot opens an mmap on the idx file + loads the cube→drafts map.
    Per-draft body parsing happens on demand via `get(draft_id)`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._jsonl_path: Path | None = None
        # Thread-local fds avoid serializing JSONL seeks; the mmap is shared.
        self._fd_local = threading.local()
        self._idx_mmap: mmap.mmap | None = None
        self._n_records = 0
        self._cube_drafts: dict[str, list[str]] = {}

    # ----- boot ------------------------------------------------------------

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._open()
            self._loaded = True

    def _open(self) -> None:
        manifest_p = TRAIN_DIR / "draft_sessions.manifest.json"
        jsonl_p = TRAIN_DIR / "draft_sessions.jsonl"
        idx_p = TRAIN_DIR / "draft_sessions.idx.bin"
        cubes_p = TRAIN_DIR / "cube_drafts_index.json"
        if not all(p.is_file() for p in (manifest_p, jsonl_p, idx_p, cubes_p)):
            log.info("Train draft sidecars missing — TrainDraftSessionsService empty.")
            return
        manifest = json.loads(manifest_p.read_text())

        # Verify the on-disk JSONL matches the hash baked into the manifest.
        # A mismatch means a partial bootstrap left a torn pair behind; refuse
        # to serve rather than return inconsistent rows.
        if not _verify_sha256(jsonl_p, manifest["jsonl_sha256"]):
            log.error(
                "Train draft_sessions.jsonl sha256 mismatch vs manifest — refusing to load. "
                "Re-run `node scripts/process_data.js --rebuild-drafts`."
            )
            return

        self._jsonl_path = jsonl_p
        self._n_records = int(manifest["n_records"])
        fd = idx_p.open("rb")
        self._idx_mmap = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)
        self._cube_drafts = json.loads(cubes_p.read_text())
        log.info(
            "Train draft sessions loaded: %d drafts, %d cubes indexed.",
            self._n_records,
            len(self._cube_drafts),
        )

    # ----- internal helpers ------------------------------------------------

    def _jsonl_fd(self):
        fd = getattr(self._fd_local, "fd", None)
        if fd is None:
            fd = self._jsonl_path.open("rb")  # type: ignore[union-attr]
            self._fd_local.fd = fd
        return fd

    def _bisect(self, draft_id: str) -> tuple[int, int] | None:
        """Return (offset, length) into the JSONL for draft_id, or None."""
        assert self._idx_mmap is not None
        target = draft_id.encode("ascii")
        if len(target) != DRAFT_ID_BYTES:
            return None
        lo, hi = 0, self._n_records
        while lo < hi:
            mid = (lo + hi) // 2
            base = mid * IDX_RECORD_BYTES
            mid_id = self._idx_mmap[base : base + DRAFT_ID_BYTES]
            if mid_id < target:
                lo = mid + 1
            elif mid_id > target:
                hi = mid
            else:
                offset = int.from_bytes(
                    self._idx_mmap[base + DRAFT_ID_BYTES : base + DRAFT_ID_BYTES + 8],
                    "little",
                )
                length = int.from_bytes(
                    self._idx_mmap[base + DRAFT_ID_BYTES + 8 : base + IDX_RECORD_BYTES],
                    "little",
                )
                return offset, length
        return None

    # ----- queries ---------------------------------------------------------

    @lru_cache(maxsize=4096)
    def get(self, draft_id: str) -> DraftSession | None:  # type: ignore[override]
        self.ensure_loaded()
        if self._idx_mmap is None:
            return None
        found = self._bisect(draft_id)
        if found is None:
            return None
        offset, length = found
        fd = self._jsonl_fd()
        fd.seek(offset)
        raw = fd.read(length).decode("utf-8")
        # Lines are formatted as: f"{draft_id}\t{json_body}\n"
        # Split once on \t — body is everything after.
        tab = raw.index("\t")
        body = json.loads(raw[tab + 1 :])
        return DraftSession(
            draft_id=draft_id,
            cube_uuid=body.get("cube_uuid"),
            owner=body.get("owner"),
            picks=body["picks"],
            synthetic=bool(body.get("synthetic", False)),
            suspect=bool(body.get("suspect", False)),
            split="train",
        )

    def list_by_cube(self, cube_uuid: str) -> list[DraftSession]:
        """Cheap to *list* (O(N_drafts_for_cube) from the dict). Each row
        still parses lazily via `get`. Callers that page through results
        only pay parse cost for the page they show."""
        self.ensure_loaded()
        ids = self._cube_drafts.get(cube_uuid, [])
        out: list[DraftSession] = []
        for did in ids:
            s = self.get(did)
            if s is not None:
                out.append(s)
        return out

    @lru_cache(maxsize=2048)
    def pick_indices_by_cube(self, cube_uuid: str) -> list[int]:  # type: ignore[override]
        """Flatten draft → picks for this cube. Cached because Phase Q's
        Collapse train-sampling does repeat lookups during a single job."""
        out: list[int] = []
        for s in self.list_by_cube(cube_uuid):
            out.extend(s.picks)
        return out

    def list_all(self, include_suspect: bool = False, limit: int = 100, offset: int = 0) -> list[DraftSession]:
        # Train is rarely browsed as "all drafts" — used only as a fallback.
        # Iterates the cube index to avoid scanning the JSONL.
        self.ensure_loaded()
        out: list[DraftSession] = []
        for ids in self._cube_drafts.values():
            for did in ids:
                s = self.get(did)
                if s is None:
                    continue
                if not include_suspect and s.suspect:
                    continue
                out.append(s)
                if len(out) >= offset + limit:
                    break
            if len(out) >= offset + limit:
                break
        return out[offset : offset + limit]


# ============================================================================
# Unified façade: routes by split, returns merged results.
# ============================================================================


class UnifiedDraftSessionsService:
    def __init__(self) -> None:
        self.val = ValDraftSessionsService()
        self.train = TrainDraftSessionsService()

    def ensure_loaded(self) -> None:
        self.val.ensure_loaded()
        self.train.ensure_loaded()

    def get(self, draft_id: str) -> DraftSession | None:
        s = self.val.get(draft_id)
        if s is not None:
            return s
        return self.train.get(draft_id)

    def list_by_cube(self, cube_uuid: str, split: Literal["train", "val", "all"] = "all") -> list[DraftSession]:
        out: list[DraftSession] = []
        if split in ("val", "all"):
            out.extend(self.val.list_by_cube(cube_uuid))
        if split in ("train", "all"):
            out.extend(self.train.list_by_cube(cube_uuid))
        return out

    def pick_indices_by_cube(self, cube_uuid: str, split: Literal["train", "val", "all"] = "all") -> list[int]:
        """Defaults to "all" so per-cube views see every indexed pick for the
        cube. Note val indices live in val_data.picks space and train indices
        in train_picks.get_pick_records space — callers that resolve records
        must dispatch by split (Phase T collapse does this)."""
        if split == "val":
            return self.val.pick_indices_by_cube(cube_uuid)
        if split == "train":
            return self.train.pick_indices_by_cube(cube_uuid)
        return list(self.val.pick_indices_by_cube(cube_uuid)) + list(self.train.pick_indices_by_cube(cube_uuid))

    def list_all(
        self,
        include_suspect: bool = False,
        limit: int = 100,
        offset: int = 0,
        split: Literal["train", "val", "all"] = "val",
    ) -> list[DraftSession]:
        if split == "val":
            return self.val.list_all(include_suspect=include_suspect, limit=limit, offset=offset)
        if split == "train":
            return self.train.list_all(include_suspect=include_suspect, limit=limit, offset=offset)
        out = self.val.list_all(include_suspect=include_suspect, limit=limit, offset=offset)
        if len(out) < limit:
            out.extend(
                self.train.list_all(
                    include_suspect=include_suspect,
                    limit=limit - len(out),
                    offset=0,
                )
            )
        return out


def _verify_sha256(path: Path, expected_hex: str) -> bool:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest() == expected_hex


draft_sessions = UnifiedDraftSessionsService()
