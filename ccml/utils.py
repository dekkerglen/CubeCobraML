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


def humanize_number(
    value: int | float,
    *,
    decimals: int = 2,
    strip_trailing_zeros: bool = True,
    suffixes: tuple[str, ...] = ("", "k", "M", "B", "T"),
) -> str:
    """Convert a large number to a compact human-readable string.

    Examples:
        >>> humanize_number(1_234)
        '1.2k'
        >>> humanize_number(7_890_000)
        '7.9M'
        >>> humanize_number(-125_000_000)
        '-125M'
        >>> humanize_number(42, decimals=0)
        '42'
    """
    if value is None or (isinstance(value, float) and (value != value)):  # NaN check
        return "NaN"

    neg = "-" if value < 0 else ""
    value = abs(value)

    if value < 1_000:
        return f"{neg}{value:.0f}"

    # Find the largest power of 1_000 that is <= the number
    magnitude = min(len(suffixes) - 1, int(len(str(int(value))) - 1) // 3)

    scaled = value / (1000**magnitude)
    fmt = f"{{:.{decimals}f}}{{}}"
    out = fmt.format(scaled, suffixes[magnitude])

    if strip_trailing_zeros and "." in out:
        out = out.rstrip("0").rstrip(".")

    return neg + out
