from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
from PIL import Image

from aoi_service.config import Config
from aoi_service.service import Service
from aoi_service.yolo_rules import Decision


class Clock:
    def __init__(self):
        self.now = 1_800_000_000.0

    def __call__(self):
        return self.now

    def advance(self, seconds=10):
        self.now += seconds


class FakePredictor:
    def __init__(self, verdicts=None, error=None):
        self.verdicts = verdicts or {}
        self.error = error
        self.calls = []
        self.crops = []

    def infer(self, source, job_id, crop):
        self.calls.append(source.name)
        self.crops.append(crop)
        if self.error:
            raise self.error
        with Image.open(source) as image:
            image.load()
            crop.pixel_box(image.size)
        ok = self.verdicts.get(source.name, True)
        return Decision(ok, "fixture OK" if ok else "fixture NG", "OK" if ok else "defect")


def write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 60), "white").save(path)


def write_xml(config, name="20261111000316_BOARD.xml", pairs=None):
    root = ET.Element("Panel", Status="FAIL", Carrier="BOARD")
    board = ET.SubElement(root, "Board", Status="FAIL", BoardSN="unchanged")
    pairs = pairs or [("top.jpg", "side.jpg")]
    for index, pair in enumerate(pairs):
        comp = ET.SubElement(board, "Component", CompName=f"C{index}", Status="False Call", Type="TEST_PART",
                             AI_result="", MachineDefect="Shift")
        images = ET.SubElement(comp, "CompImage")
        ET.SubElement(images, "Image", PicPath1=f"old/path/{pair[0]}",
                      PicPath2=f"old/path/{pair[1]}", Angle="90",
                      X1="10", Y1="10", X2="70", Y2="50")
        ET.SubElement(comp, "Window", AI_result="unchanged", PicPath1="ignored.jpg")
    source = config.paths.xml_input / name
    source.write_bytes(ET.tostring(root))
    return source


@pytest.fixture
def config(tmp_path):
    settings = Config.model_validate({
        "watch": {"stable_seconds": 0, "poll_seconds": 1, "retry_seconds": 1,
                  "image_wait_timeout_seconds": 30, "max_retries": 2}
    }).resolve_paths(tmp_path)
    settings.paths.bom_list.parent.mkdir(parents=True, exist_ok=True)
    settings.paths.bom_list.write_text("TEST_PART\n", encoding="utf-8")
    settings.prepare(require_model=False)
    return settings


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def service_factory(config, clock):
    services = []

    def make(predictor=None, settings=None):
        service = Service(settings or config, predictor or FakePredictor(), clock)
        services.append(service)
        return service

    yield make
    for service in services:
        if not getattr(service, "test_closed", False):
            service.close()


def close_service(service):
    service.close()
    service.test_closed = True
