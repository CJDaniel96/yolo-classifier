"""One persistent Ultralytics OBB model, using the supplied CLI's rules."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from .config import Config
from .regions import CropBox
from .yolo_rules import (
    Decision, ObbRules, classify_detections, detections_from_result,
    normalize_names, resolve_ok_class_id, save_review_artifacts, unique_output_targets,
)


class Predictor:
    def __init__(self, config: Config):
        from ultralytics import YOLO

        self.config = config
        self.model = YOLO(str(config.inference.model))
        if self.model.task != "obb":
            raise ValueError("The configured model must be a YOLO OBB model")
        self.names = normalize_names(self.model.names)
        self.ok_id = resolve_ok_class_id(self.names, config.inference.ok_label)
        settings = config.inference
        self.rules = ObbRules(
            angle_target_deg=settings.ok_angle_target,
            angle_tolerance_deg=settings.ok_angle_tolerance,
            max_center_distance_px=settings.ok_max_center_distance,
            min_edge_distance_px=settings.ok_min_edge_distance,
        )

    def infer(self, source: Path, job_id: str, crop: CropBox) -> Decision:
        settings = self.config.inference
        with Image.open(source) as image:
            image.load()
            frame = image.crop(crop.pixel_box(image.size)).convert("RGB")
        options = dict(conf=settings.conf, iou=settings.iou, imgsz=settings.imgsz,
                       max_det=settings.max_det, verbose=False)
        if settings.device is not None:
            options["device"] = settings.device
        results = self.model.predict(source=frame, **options)
        if not results or getattr(results[0], "obb", None) is None:
            raise RuntimeError("The model did not return an OBB result")
        result = results[0]
        detections = detections_from_result(result, self.names)
        height, width = map(int, result.orig_shape)
        if (width, height) != frame.size:
            raise RuntimeError("The model's original shape does not match the cropped image")
        decision = classify_detections(
            detections, self.ok_id, settings.ok_count, (width, height),
            settings.center_tolerance, settings.ok_overlap_iou, self.rules,
        )
        if self.config.review.enabled and not self.config.runtime.dry_run:
            directory = self.config.review.output_dir / job_id / f"roi-{crop.identifier}"
            directory = directory / ("OK" if decision.is_ok else "NG")
            if not decision.is_ok:
                directory /= decision.folder
            targets = unique_output_targets(
                directory, source.name, set(), self.config.review.on_exists,
                include_measurements=True,
            )
            if targets is not None:
                # The inherited saver copies its source and emits crop-relative VOC boxes.
                # Supply the actual cropped frame so the review image and annotations match.
                with TemporaryDirectory(prefix="aoi-roi-") as temporary:
                    cropped_source = Path(temporary) / source.name
                    frame.save(cropped_source)
                    save_review_artifacts(result, cropped_source, targets, detections,
                                          (width, height), decision)
        return decision
