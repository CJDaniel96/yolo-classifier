"""Load ChipRC part numbers as an exact-match allowlist."""

from pathlib import Path


def load_bom(path: Path) -> frozenset[str]:
    try:
        contents = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read BOM list {path}: {exc}") from exc
    parts = frozenset(line.strip() for line in contents.splitlines() if line.strip())
    if not parts:
        raise ValueError(f"BOM list contains no part numbers: {path}")
    return parts
