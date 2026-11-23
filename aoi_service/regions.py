"""Required pixel rectangles and per-file inference identities."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class CropBox:
    """Original-image coordinates: left/top inclusive, right/bottom exclusive."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Image crop coordinates must all be finite numbers")
        if self.x1 < 0 or self.y1 < 0:
            raise ValueError("Image crop X1 and Y1 must be nonnegative")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Image crop requires X2 > X1 and Y2 > Y1")

    @classmethod
    def from_attributes(cls, attributes: Mapping[str, str]) -> CropBox:
        values = []
        for name in ("X1", "Y1", "X2", "Y2"):
            raw = attributes.get(name, "").strip()
            if not raw:
                raise ValueError(f"Image crop coordinate {name} is missing or empty")
            try:
                values.append(float(raw))
            except ValueError as exc:
                raise ValueError(f"Image crop coordinate {name} is not a number: {raw!r}") from exc
        return cls(*values)

    def pixel_box(self, size: tuple[int, int]) -> tuple[int, int, int, int]:
        width, height = size
        if self.x2 > width or self.y2 > height:
            raise ValueError(
                f"Image crop {self.as_tuple()} exceeds image bounds {width}x{height}"
            )
        # Include the complete fractional rectangle without Pillow's implicit rounding.
        return (math.floor(self.x1), math.floor(self.y1),
                math.ceil(self.x2), math.ceil(self.y2))

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def identifier(self) -> str:
        encoded = json.dumps([0.0 if value == 0 else float(value) for value in self.as_tuple()])
        return hashlib.sha256(encoded.encode()).hexdigest()[:20]


@dataclass(frozen=True)
class ImageRegion:
    filename: str
    crop: CropBox

    @property
    def key(self) -> str:
        return f"{self.filename}#roi-{self.crop.identifier}"
