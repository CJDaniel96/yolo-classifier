"""File integrity, Windows-safe names, and a cross-platform process lock."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def image_basename(value: str) -> str:
    name = value.replace("\\", "/").rsplit("/", 1)[-1]
    if (not name or name in (".", "..") or name[-1:] in (".", " ")
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or name.split(".")[0].upper() in {
                "CON", "PRN", "AUX", "NUL",
                *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10)),
            }):
        raise ValueError(f"Invalid image filename in XML: {value!r}")
    return name


def archive_folder(filename: str, root: Path) -> Path:
    match = re.fullmatch(r"(\d{14})_(.+)\.[xX][mM][lL]", filename)
    if not match:
        raise ValueError("XML filename must be timestamp_BoardSN.xml")
    stamp = match[1]
    date = datetime.strptime(stamp, "%Y%m%d%H%M%S")
    return root / date.strftime("%Y-%m-%d") / stamp


def direct_file(path: Path, root: Path) -> None:
    """Never follow input symlinks outside the configured flat input folder."""
    if path.is_symlink() or path.resolve().parent != root.resolve():
        raise ValueError(f"Input must be a direct regular file: {path}")


def atomic_bytes(path: Path, data: bytes) -> None:
    """Publish a complete new file without overwriting an existing destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "nt":
            os.rename(temporary, path)  # Windows rename refuses an existing target.
        else:
            os.link(temporary, path)  # POSIX rename would overwrite the target.
    finally:
        temporary.unlink(missing_ok=True)


def verified_copy(source: Path, target: Path, expected: str) -> None:
    """Idempotent copy. A different existing archive is never overwritten."""
    if target.exists():
        if target.is_symlink() or digest(target) != expected:
            raise FileExistsError(f"Archive collision (different content): {target}")
        return
    if source.is_symlink() or digest(source) != expected:
        raise RuntimeError(f"Source changed before archive: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with source.open("rb") as src, temporary.open("xb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        if digest(temporary) != expected:
            raise RuntimeError(f"Copy integrity check failed: {source}")
        # The service has a process lock. Recheck to avoid overwriting external files.
        if target.exists():
            if digest(target) != expected:
                raise FileExistsError(f"Archive collision: {target}")
        else:
            # Exclusive creation also protects against races with external writers.
            try:
                if os.name == "nt":
                    os.rename(temporary, target)
                else:
                    os.link(temporary, target)
            except FileExistsError:
                if digest(target) != expected:
                    raise
    finally:
        temporary.unlink(missing_ok=True)


class ProcessLock:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self.stream.seek(0, os.SEEK_END)
                if self.stream.tell() == 0:
                    self.stream.write(b"0")
                    self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise RuntimeError("Another service instance is using this state database") from exc

    def close(self) -> None:
        self.stream.close()
