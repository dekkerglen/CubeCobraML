"""Read and cache training metrics.csv files.

Re-reads on disk-mtime change so the UI sees new epoch rows without a server
restart. Cheap — the CSVs are <1 MB.
"""

from __future__ import annotations

import csv
import threading
from pathlib import Path

from backend.lib.config import RUNS_DIR


class MetricsService:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, list[dict]]] = {}
        self._lock = threading.Lock()

    def list_runs(self) -> list[str]:
        if not RUNS_DIR.is_dir():
            return []
        return sorted(
            (p.name for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "metrics.csv").exists()),
            reverse=True,
        )

    def latest_run(self) -> str | None:
        runs = self.list_runs()
        return runs[0] if runs else None

    def rows(self, run_id: str) -> list[dict]:
        path = RUNS_DIR / run_id / "metrics.csv"
        if not path.exists():
            return []
        mtime = path.stat().st_mtime
        with self._lock:
            cached = self._cache.get(run_id)
            if cached and cached[0] == mtime:
                return cached[1]
        rows = self._read(path)
        with self._lock:
            self._cache[run_id] = (mtime, rows)
        return rows

    @staticmethod
    def _read(path: Path) -> list[dict]:
        out: list[dict] = []
        with path.open() as fh:
            for raw in csv.DictReader(fh):
                row: dict[str, float | None] = {}
                for k, v in raw.items():
                    if v in (None, ""):
                        row[k] = None
                    else:
                        try:
                            row[k] = float(v)
                        except ValueError:
                            row[k] = None
                out.append(row)
        return out

    def last_completed_row(self, run_id: str) -> dict | None:
        """The most recent row where val_bomb_agreement is non-null —
        i.e. a row that includes validation, not just train."""
        rows = self.rows(run_id)
        for r in reversed(rows):
            if r.get("val_bomb_agreement") is not None:
                return r
        return rows[-1] if rows else None


metrics_service = MetricsService()
