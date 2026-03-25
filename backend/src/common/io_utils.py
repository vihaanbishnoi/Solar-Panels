"""I/O helpers shared across tasks."""

from pathlib import Path


def ensure_parent_dir(path: Path) -> None:
    """Create parent directory for a file path if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)

