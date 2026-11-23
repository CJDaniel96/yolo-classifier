from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from aoi_service.api import create_app
from aoi_service.inference import Predictor
from aoi_service.regions import CropBox
from conftest import FakePredictor, write_image


def test_http_model_loaded_once_and_worker_stops(config):
    factories = []

    def factory(settings):
        factories.append(settings)
        return FakePredictor()

    app = create_app(config, factory)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/status").json()["worker_alive"] is True
        assert client.get("/jobs").json() == []
        assert client.get("/jobs?limit=10000").status_code == 422
        assert len(factories) == 1
        app.state.service.last_error = "network unavailable"
        assert client.get("/health").status_code == 503
    assert not app.state.service.thread.is_alive()


def test_second_instance_is_rejected_before_model_load(config, service_factory):
    service_factory()
    loaded = []
    from aoi_service.service import Service
    with pytest.raises(RuntimeError, match="Another service instance"):
        Service(config, lambda: loaded.append(True))
    assert loaded == []


class FakeObb:
    cls = np.array([0])
    conf = np.array([0.95])
    xyxy = np.array([[20, 15, 40, 25]])
    xyxyxyxy = np.array([[[20, 15], [40, 15], [40, 25], [20, 25]]])

    def cpu(self):
        return self

    def __len__(self):
        return 1


def test_real_adapter_options_obb_rules_and_review_artifacts(config, monkeypatch):
    result = SimpleNamespace(obb=FakeObb(), orig_shape=(40, 60),
                             plot=lambda: np.zeros((40, 60, 3), dtype=np.uint8))
    calls = []

    class Model:
        names = {0: "OK", 1: "defect"}
        task = "obb"

        def __init__(self, model):
            assert model == str(config.inference.model)

        def predict(self, **options):
            calls.append(options)
            return [result]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    config.review.enabled = True
    config.inference.device = 0
    config.inference.ok_count = 1
    config.inference.ok_angle_tolerance = 5
    source = config.paths.image_input / "top.jpg"
    write_image(source)
    predictor = Predictor(config)
    crop = CropBox(10, 10, 70, 50)
    decision = predictor.infer(source, "job123", crop)
    assert decision.is_ok
    assert calls[0]["device"] == 0
    assert calls[0]["conf"] == config.inference.conf
    assert calls[0]["imgsz"] == config.inference.imgsz
    assert calls[0]["source"].size == (60, 40)
    directory = config.review.output_dir / "job123" / f"roi-{crop.identifier}" / "OK"
    assert (directory / "inference/top.jpg").exists()
    assert (directory / "original/top.xml").exists()
    data = json.loads((directory / "measurements/top.json").read_text())
    assert data["verdict"] == "OK"
    assert data["measurements"]["effective_ok_count"] == 1
    assert data["measurements"]["image_size"] == [60, 40]
    from PIL import Image
    with Image.open(directory / "original/top.jpg") as image:
        assert image.size == (60, 40)


def test_adapter_rejects_non_obb_model(config, monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(
        YOLO=lambda _: SimpleNamespace(task="detect", names={0: "OK"})))
    with pytest.raises(ValueError, match="OBB"):
        Predictor(config)
