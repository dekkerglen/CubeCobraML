import subprocess
from pathlib import Path


def _get_git_root() -> Path:
    try:
        git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL)
        return Path(git_root.decode().strip())
    except subprocess.CalledProcessError:
        raise RuntimeError("Not inside a Git repository.")


ROOT = _get_git_root()
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
