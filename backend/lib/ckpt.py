"""Checkpoint discovery + loading for the dashboard.

A "checkpoint" in the dashboard is one of three things:
  - a `runs/<run_id>/ckpts/step_..._epoch_NNN` directory written by the
    PeriodicSnapshot callback during training
  - the always-on `runs/<run_id>/ckpts/latest` directory written every epoch
  - `runs/<run_id>/ckpts/best` symlink to the highest-metric ckpt
  - the standalone repo-root `model/` from training's final save
  - the prod TFJS model (loaded via tfjs_load)

All loaders return objects that quack like `CubeCobraMLSystem` for the heads
we expose to the dashboard. Prod uses standalone Sequential models per head
loaded by `tfjs_load`; current/run ckpts use the project's CubeCobraMLSystem.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# Allow `from ccml.model import ...` from anywhere the dashboard runs.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from .config import MODEL_DIR, RUNS_DIR, prod_model_path  # noqa: E402

PROD_KEY = "prod"
CURRENT_KEY = "current"

# Subdirs the current CubeCobraMLSystem.load_weights expects. A ckpt is only
# offered to the picker if all of these exist. Older ckpt dirs with extra
# subdirs (e.g. delta_projection from the ctx-era architecture) still pass —
# the extras are simply ignored by load_weights.
REQUIRED_HEADS = (
    "encoder",
    "cube_decoder",
    "draft_decoder",
    "deck_build_decoder",
    "correlation_decoder",
)


def _ckpt_complete(path: Path) -> bool:
    """True iff `path` contains all required head subdirs with a model file
    inside each (Keras 3 saves as `model.keras` under each head's subdir)."""
    if not path.is_dir():
        return False
    for head in REQUIRED_HEADS:
        head_dir = path / head
        if not head_dir.is_dir():
            return False
        # Keras 3 saves as `model.keras`; some older builds save as `model` index
        # files. Accept either.
        if not any(head_dir.iterdir()):
            return False
    return True


@dataclass(slots=True, frozen=True)
class CkptEntry:
    """One selectable checkpoint."""

    key: str
    label: str
    path: Path | None
    run_id: str | None
    epoch: int | None
    metric: float | None
    kind: str  # "ckpt", "latest", "best", "current", "prod"


@dataclass(slots=True, frozen=True)
class Run:
    """A training run as a first-class entity (one experiment).

    A run aggregates all the ckpts under `runs/<run_id>/ckpts/` plus the
    metadata in `runs/<run_id>/meta.json`. The dashboard uses this to drive
    the runs-first model selector: pick a run, then optionally drill into
    best/latest/specific ckpt under it.
    """

    run_id: str
    path: Path
    label: str  # human display name (meta.label > derived > short)
    short: str  # "Jun 06 17:51 · 73f840d4"
    start_utc: str  # ISO timestamp from meta.json
    git_sha: str
    archived: bool
    best_key: str | None  # ckpt key of the best ckpt under this run
    latest_key: str | None  # ckpt key of the always-on latest
    best_metric: float | None
    best_mode: str  # "min" | "max"
    best_metric_name: str
    n_ckpts: int


# ----- run-level metadata helpers -------------------------------------------


# Env vars worth showing in the auto-derived label, in display order.
# Pairs of (env var, render fn). Render fn returns "" to skip silently.
def _fmt_lr(val: str) -> str:
    try:
        f = float(val)
    except ValueError:
        return ""
    # Render 3e-4, 1e-5 etc. compactly.
    return f"{f:.0e}".replace("e-0", "e-").replace("e+0", "e+")


_LABEL_ENV_ORDER: tuple[tuple[str, str], ...] = (
    # (env var, label prefix)
    ("LR_PEAK", "lr"),
    ("HIST_LOSS_WEIGHT", "hist"),
    ("PACK_WEIGHT_POWER", "wPow"),
    ("DRAFT_BIAS_ALPHA", "α"),
    ("PICK_LABEL_SMOOTHING", "ε"),
)


def _derive_label_from_env(env: dict) -> str:
    """One-line summary of a run's hyperparameters from meta.env."""
    parts: list[str] = []
    for var, prefix in _LABEL_ENV_ORDER:
        if var not in env:
            continue
        raw = str(env[var])
        if var == "LR_PEAK":
            end = env.get("LR_END")
            lr_peak = _fmt_lr(raw)
            lr_end = _fmt_lr(str(end)) if end is not None else ""
            if lr_peak and lr_end:
                parts.append(f"{prefix} {lr_peak}→{lr_end}")
            elif lr_peak:
                parts.append(f"{prefix} {lr_peak}")
        elif var == "HIST_LOSS_WEIGHT":
            parts.append(f"{prefix}={_fmt_lr(raw) or raw}")
        else:
            parts.append(f"{prefix}={raw}")
    return " · ".join(parts)


def _short_run_label(start_utc: str, git_sha: str) -> str:
    """Compact 'Jun 06 17:51 · 73f840d4' style label from start_utc + git_sha."""
    try:
        dt = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
        date_str = dt.strftime("%b %d %H:%M")
    except (ValueError, AttributeError):
        date_str = start_utc[:16] if start_utc else "?"
    sha = (git_sha or "")[:8]
    return f"{date_str} · {sha}" if sha else date_str


def _read_meta(run_dir: Path) -> dict:
    """Read meta.json safely. Returns {} on any error."""
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _resolve_run_label(meta: dict, short: str) -> str:
    """Label resolution: meta.label > derived from env > short."""
    if (manual := meta.get("label")) and isinstance(manual, str) and manual.strip():
        return manual.strip()
    env = meta.get("env", {}) or {}
    if env:
        derived = _derive_label_from_env(env)
        if derived:
            return derived
    return short


def list_runs() -> list[Path]:
    """All discoverable runs, sorted by run_id descending (newest first).

    Looks under both `runs/<id>/` (active/experiments) and `runs/archive/<id>/`
    (manually archived). A run is recognized by the presence of a `ckpts/`
    subdir — the archive/ directory itself is skipped because it has no ckpts.
    """
    if not RUNS_DIR.is_dir():
        return []
    flat = [p for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "ckpts").is_dir()]
    archive_dir = RUNS_DIR / "archive"
    archived: list[Path] = []
    if archive_dir.is_dir():
        archived = [p for p in archive_dir.iterdir() if p.is_dir() and (p / "ckpts").is_dir()]
    # Sort by dir name (timestamped) descending so newest is first.
    return sorted(flat + archived, key=lambda p: p.name, reverse=True)


def list_ckpts(run_dir: Path) -> list[CkptEntry]:
    """All saved ckpts in a single run dir, in epoch order."""
    out: list[CkptEntry] = []
    ckpts_dir = run_dir / "ckpts"
    index_path = ckpts_dir / "best_index.json"
    metrics: dict[str, float] = {}
    if index_path.exists():
        try:
            d = json.loads(index_path.read_text())
            metrics = {r["dirname"]: r["metric"] for r in d.get("ranked", [])}
        except Exception:
            pass

    for child in sorted(ckpts_dir.iterdir()) if ckpts_dir.exists() else []:
        if not child.is_dir():
            continue
        if child.name == "best":
            continue  # skip the symlink, already shown as the linked target
        if not _ckpt_complete(child):
            continue  # skip ckpts missing any required head subdir
        if child.name == "latest":
            meta_path = child / "_meta.json"
            ep = None
            if meta_path.exists():
                try:
                    ep = json.loads(meta_path.read_text()).get("epoch")
                except Exception:
                    pass
            out.append(
                CkptEntry(
                    key=f"{run_dir.name}/latest",
                    label=f"{run_dir.name} • latest" + (f" (ep {ep})" if ep is not None else ""),
                    path=child,
                    run_id=run_dir.name,
                    epoch=ep,
                    metric=None,
                    kind="latest",
                )
            )
            continue
        epoch = None
        if child.name.startswith("step_") and "_epoch_" in child.name:
            try:
                epoch = int(child.name.split("_epoch_")[1])
            except ValueError:
                pass
        out.append(
            CkptEntry(
                key=f"{run_dir.name}/{child.name}",
                label=f"{run_dir.name} • ep {epoch} • {metrics.get(child.name, 0):.4f}",
                path=child,
                run_id=run_dir.name,
                epoch=epoch,
                metric=metrics.get(child.name),
                kind="ckpt",
            )
        )
    out.sort(key=lambda e: e.epoch if e.epoch is not None else -1)
    return out


def all_selectable() -> list[CkptEntry]:
    """Every checkpoint the dashboard can offer in a dropdown."""
    out: list[CkptEntry] = []
    if _ckpt_complete(MODEL_DIR):
        out.append(
            CkptEntry(
                key=CURRENT_KEY,
                label="model/ (current top-level save)",
                path=MODEL_DIR,
                run_id=None,
                epoch=None,
                metric=None,
                kind="current",
            )
        )
    prod = prod_model_path()
    if prod is not None:
        out.append(
            CkptEntry(
                key=PROD_KEY,
                label=f"prod TFJS ({prod})",
                path=prod,
                run_id=None,
                epoch=None,
                metric=None,
                kind="prod",
            )
        )
    for run in list_runs():
        out.extend(list_ckpts(run))
    return out


def find_best(entries: list[CkptEntry]) -> CkptEntry | None:
    """Highest val_bomb_agreement entry, if any."""
    scored = [e for e in entries if e.metric is not None]
    if not scored:
        return None
    return max(scored, key=lambda e: e.metric)


# ----- Run-level discovery (the runs-first selector API) ---------------------


def _run_archive_status(path: Path) -> bool:
    """True iff this run dir lives under runs/archive/."""
    return path.parent.name == "archive"


def _run_keys(run_dir: Path, ckpts: list[CkptEntry]) -> tuple[str | None, str | None]:
    """Return (best_key, latest_key) for the run, by scanning best_index.json
    + the latest ckpt (which has its own special CkptEntry from list_ckpts).
    """
    latest_key: str | None = None
    for c in ckpts:
        if c.kind == "latest":
            latest_key = c.key
            break

    best_key: str | None = None
    index_path = run_dir / "ckpts" / "best_index.json"
    if index_path.exists():
        try:
            d = json.loads(index_path.read_text())
            ranked = d.get("ranked", [])
            if ranked:
                best_dirname = ranked[0]["dirname"]
                best_key = f"{run_dir.name}/{best_dirname}"
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    # If best_index missing/broken, fall back to highest-metric ckpt locally.
    if best_key is None:
        scored = [c for c in ckpts if c.metric is not None]
        if scored:
            best_key = max(scored, key=lambda e: e.metric).key
    return best_key, latest_key


def _best_metric(run_dir: Path) -> tuple[float | None, str, str]:
    """(value, mode, metric_name) from best_index.json. Defaults if unreadable."""
    index_path = run_dir / "ckpts" / "best_index.json"
    if not index_path.exists():
        return None, "max", "val_bomb_agreement"
    try:
        d = json.loads(index_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None, "max", "val_bomb_agreement"
    mode = d.get("best_mode", "max")
    name = d.get("best_metric", "val_bomb_agreement")
    ranked = d.get("ranked", [])
    val = float(ranked[0]["metric"]) if ranked else None
    return val, mode, name


def list_runs_rich() -> list[Run]:
    """Every discoverable run as a `Run` dataclass.

    Sort order: non-archived first (newest first), then archived (newest first).
    """
    out: list[Run] = []
    for run_dir in list_runs():
        ckpts = list_ckpts(run_dir)
        if not ckpts:
            continue
        meta = _read_meta(run_dir)
        start_utc = meta.get("start_utc", "")
        git_sha = meta.get("git_sha", "")
        short = _short_run_label(start_utc, git_sha)
        label = _resolve_run_label(meta, short)
        best_key, latest_key = _run_keys(run_dir, ckpts)
        best_metric, best_mode, best_metric_name = _best_metric(run_dir)
        out.append(
            Run(
                run_id=run_dir.name,
                path=run_dir,
                label=label,
                short=short,
                start_utc=start_utc,
                git_sha=git_sha,
                archived=_run_archive_status(run_dir),
                best_key=best_key,
                latest_key=latest_key,
                best_metric=best_metric,
                best_mode=best_mode,
                best_metric_name=best_metric_name,
                n_ckpts=len(ckpts),
            )
        )
    # Group: non-archived first (each group already in newest-first order
    # because list_runs sorts by name desc).
    return sorted(out, key=lambda r: (r.archived,))


def find_most_recent_run(runs: list[Run]) -> Run | None:
    """Newest non-archived run by start_utc; falls back to newest overall."""
    non_archived = [r for r in runs if not r.archived]
    pool = non_archived or runs
    if not pool:
        return None
    return max(pool, key=lambda r: r.start_utc or "")


# ----- Model loaders ---------------------------------------------------------


def load_system(path_str: str):
    """Load a CubeCobraMLSystem from a directory (current or runs/ ckpt).

    Callers should cache the result (see `backend.services.model_registry`)
    — load is ~3-5 s.
    """
    from ccml.model import CubeCobraMLSystem

    from .cards import num_cards

    sys_model = CubeCobraMLSystem(num_cards())
    sys_model.load_weights(path_str)
    return sys_model


def load_prod(prod_path_str: str):
    """Load all prod heads as a ProdModel that exposes CubeCobraMLSystem's
    API. Passes the current pipeline's oracle list so the wrapper can detect
    index-space drift if the user reprocessed raw_data after prod was exported.
    """
    from .cards import _oracle_index
    from .tfjs_load import load_prod_model

    return load_prod_model(prod_path_str, current_oracle_list=_oracle_index())


def load_entry(entry: CkptEntry):
    """Dispatch to the right loader. Returns an object exposing
    `.draft / .deck_build / .recommend / .encoder`
    — `CubeCobraMLSystem` for trained ckpts, `ProdModel` wrapper for prod.
    Pages don't need to know which.
    """
    if entry.kind == "prod":
        return load_prod(str(entry.path))
    return load_system(str(entry.path))
