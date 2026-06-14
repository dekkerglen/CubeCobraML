"""Path resolution for the dashboard.

All paths derive from the repo root (auto-detected from this file's location)
or from environment variables. Nothing user-specific is hardcoded.

Env vars:
    CUBECOBRA_PROD_MODEL  - directory containing the prod TFJS heads
                            (encoder/, draft_decoder/, ...). Typically points
                            at <CubeCobra checkout>/packages/recommenderService/model.
                            If unset, dashboard skips prod-model loading.

The dashboard also auto-loads `<repo>/.env` at import time (simple `KEY=VALUE`
lines, no quoting, no escapes), so you can persist `CUBECOBRA_PROD_MODEL`
without exporting it in every shell session. Existing shell-set vars take
precedence over `.env`.
"""

from __future__ import annotations

import os
from pathlib import Path


# This file: <repo>/dashboard/lib/config.py → REPO_ROOT is two parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
RUNS_DIR = REPO_ROOT / "runs"
MODEL_DIR = REPO_ROOT / "model"
RAW_DATA_DIR = REPO_ROOT / "raw_data"


def _load_dotenv() -> None:
    """Load `<repo>/.env` into os.environ if present. Shell vars win."""
    dotenv = REPO_ROOT / ".env"
    if not dotenv.is_file():
        return
    for raw in dotenv.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv()


def prod_model_path() -> Path | None:
    """Resolve the prod model location from CUBECOBRA_PROD_MODEL env var.

    Returns None if the env var is unset or the path doesn't exist. The
    dashboard pages should handle None gracefully (just don't offer prod as a
    selectable checkpoint).
    """
    env = os.environ.get("CUBECOBRA_PROD_MODEL")
    if not env:
        return None
    p = Path(env).expanduser().resolve()
    if not p.is_dir():
        return None
    if not any((p / head / "model.json").exists() for head in ("encoder", "draft_decoder", "cube_decoder")):
        return None
    return p


def prod_model_hint() -> str:
    """Human-readable hint for users who haven't set CUBECOBRA_PROD_MODEL."""
    return (
        "Set CUBECOBRA_PROD_MODEL to your CubeCobra checkout's "
        "packages/recommenderService/model directory, e.g.\n"
        "  export CUBECOBRA_PROD_MODEL=~/Projects/CubeCobra/packages/recommenderService/model"
    )
