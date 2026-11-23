"""Validated JSON configuration; relative paths are relative to the JSON file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)


class Paths(Settings):
    xml_input: Path = Path("input/xml")
    image_input: Path = Path("input/images")
    xml_output: Path = Path("output")
    archive_dir: Path | None = None
    state_db: Path = Path("runtime/jobs.sqlite3")
    log_dir: Path = Path("logs")


class Inference(Settings):
    model: Path = Path("models/best.pt")
    ok_label: str = Field(default="OK", min_length=1)
    ok_count: StrictInt | None = Field(default=None, ge=1)
    ok_overlap_iou: float = Field(default=0.8, ge=0, le=1)
    center_tolerance: float = Field(default=0.25, ge=0, le=0.5)
    ok_angle_target: float = 0.0
    ok_angle_tolerance: float | None = Field(default=None, ge=0, le=90)
    ok_max_center_distance: float | None = Field(default=None, ge=0)
    ok_min_edge_distance: float | None = Field(default=None, ge=0)
    conf: float = Field(default=0.25, ge=0, le=1)
    iou: float = Field(default=0.7, ge=0, le=1)
    imgsz: StrictInt = Field(default=640, ge=1)
    device: str | StrictInt | None = None
    max_det: StrictInt = Field(default=300, ge=1)
    ext: list[str] = Field(default_factory=lambda: [
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
    ], min_length=1)


class Watch(Settings):
    poll_seconds: float = Field(default=2, gt=0)
    stable_seconds: float = Field(default=2, ge=0)
    image_wait_timeout_seconds: float = Field(default=300, gt=0)
    retry_seconds: float = Field(default=10, gt=0)
    max_retries: StrictInt = Field(default=3, ge=1)
    failure_statuses: list[str] = Field(default_factory=lambda: ["FAIL", "NG"], min_length=1)


class Archive(Settings):
    after_success: Literal["archive", "delete", "keep"] = "archive"
    retention_days: StrictInt | None = Field(default=None, ge=1)


class Review(Settings):
    enabled: StrictBool = False
    output_dir: Path = Path("review")
    on_exists: Literal["suffix", "skip", "overwrite"] = "suffix"


class Runtime(Settings):
    dry_run: StrictBool = False
    verbose: StrictBool = False
    quiet: StrictBool = False


class Http(Settings):
    host: str = "127.0.0.1"
    port: StrictInt = Field(default=8000, ge=1, le=65535)


class Config(Settings):
    paths: Paths = Field(default_factory=Paths)
    inference: Inference = Field(default_factory=Inference)
    watch: Watch = Field(default_factory=Watch)
    archive: Archive = Field(default_factory=Archive)
    review: Review = Field(default_factory=Review)
    runtime: Runtime = Field(default_factory=Runtime)
    http: Http = Field(default_factory=Http)

    @model_validator(mode="after")
    def coherent(self) -> Config:
        if self.runtime.verbose and self.runtime.quiet:
            raise ValueError("runtime.verbose and runtime.quiet cannot both be true")
        if any(not s.strip() for s in self.watch.failure_statuses):
            raise ValueError("watch.failure_statuses must contain nonempty status names")
        return self

    def resolve_paths(self, base: Path) -> Config:
        def absolute(path: Path) -> Path:
            return (base / path).resolve()

        for key in ("xml_input", "image_input", "xml_output", "state_db", "log_dir"):
            setattr(self.paths, key, absolute(getattr(self.paths, key)))
        self.paths.archive_dir = absolute(
            self.paths.archive_dir or self.paths.xml_input / "DONE"
        )
        self.inference.model = absolute(self.inference.model)
        self.review.output_dir = absolute(self.review.output_dir)
        self.inference.ext = [
            (e if e.startswith(".") else "." + e).lower() for e in self.inference.ext
        ]
        inputs = (self.paths.xml_input, self.paths.image_input)
        if inputs[0] == inputs[1]:
            raise ValueError("XML and image input folders must be different")
        managed = (self.paths.xml_output, self.paths.archive_dir, self.review.output_dir)
        if any(path in inputs for path in managed):
            raise ValueError("Output/archive/review folders must differ from input folders")
        if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
               for i, a in enumerate(managed) for b in managed[i + 1:]):
            raise ValueError("Output/archive/review folders must not overlap")
        if self.paths.xml_output.is_relative_to(self.paths.xml_input):
            raise ValueError("XML output must be outside XML input")
        if any(i.is_relative_to(m) for i in inputs for m in managed):
            raise ValueError("Input folders must not be inside output/archive/review folders")
        return self

    def prepare(self, require_model: bool = True) -> None:
        if require_model and not self.inference.model.is_file():
            raise ValueError(f"Model file not found: {self.inference.model}")
        for directory in (self.paths.xml_input, self.paths.image_input,
                          self.paths.xml_output, self.paths.log_dir,
                          self.paths.state_db.parent):
            directory.mkdir(parents=True, exist_ok=True)
        if self.archive.after_success == "archive" and not self.runtime.dry_run:
            self.paths.archive_dir.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    return Config.model_validate(raw).resolve_paths(path.resolve().parent)
