from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from aoi_service.files import archive_folder, digest
from aoi_service.inference import Predictor
from aoi_service.regions import CropBox
from aoi_service.xmlio import Document
from conftest import FakePredictor, close_service, write_image, write_xml


class OneObb:
    def __init__(self, width, height, ok):
        self.cls = np.array([0 if ok else 1])
        self.conf = np.array([0.95])
        x1, x2 = width * 0.25, width * 0.75
        y1, y2 = height * 0.4, height * 0.6
        self.xyxy = np.array([[x1, y1, x2, y2]])
        self.xyxyxyxy = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]])

    def cpu(self):
        return self

    def __len__(self):
        return 1


def patch_model(monkeypatch, calls, decide=lambda frame: True):
    class Model:
        names = {0: "OK", 1: "defect"}
        task = "obb"

        def __init__(self, model):
            pass

        def predict(self, source, **kwargs):
            calls.append(source.copy())
            width, height = source.size
            return [SimpleNamespace(obb=OneObb(width, height, decide(source)),
                                    orig_shape=(height, width))]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))


@pytest.mark.parametrize("values", [
    {"X2": ""}, {"Y2": "  "}, {"X1": "not-a-number"},
    {"X1": "NaN"}, {"X2": "Infinity"}, {"Y1": "-Infinity"},
    {"X1": "-1"}, {"Y1": "-1"}, {"X2": "10"}, {"Y2": "10"},
    {"X2": "9"}, {"Y2": "9"},
])
def test_missing_nonfinite_or_reversed_coordinates_rejected(values):
    attributes = dict(X1="10", Y1="10", X2="70", Y2="50")
    attributes.update(values)
    with pytest.raises(ValueError):
        CropBox.from_attributes(attributes)


def test_coordinate_attribute_must_exist():
    with pytest.raises(ValueError, match="X2"):
        CropBox.from_attributes(dict(X1="1", Y1="2", Y2="40"))


def test_fractional_roi_reaches_model_as_exact_pixels(config, monkeypatch):
    calls = []
    patch_model(monkeypatch, calls)
    config.inference.ok_max_center_distance = 0
    # A gradient proves the origin as well as the cropped dimensions.
    pixels = np.zeros((60, 80, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(80)
    pixels[:, :, 1] = np.arange(60)[:, None]
    source = config.paths.image_input / "gradient.png"
    Image.fromarray(pixels).save(source)
    original_hash = digest(source)
    crop = CropBox(10.2, 20.8, 30.1, 40.2)
    result = Predictor(config).infer(source, "job", crop)
    assert result.is_ok  # Geometric limits use the cropped image's centre.
    assert len(calls) == 1
    assert calls[0].size == (21, 21)
    assert calls[0].getpixel((0, 0)) == (10, 20, 0)
    assert calls[0].getpixel((20, 20)) == (30, 40, 0)
    assert result.measurements.image_size == (21, 21)
    assert result.measurements.objects[0].center_distance_px == 0
    assert digest(source) == original_hash


@pytest.mark.parametrize("box", [CropBox(0, 0, 81, 60), CropBox(0, 0, 80, 61)])
def test_out_of_bounds_never_calls_model(config, monkeypatch, box):
    calls = []
    patch_model(monkeypatch, calls)
    source = config.paths.image_input / "top.jpg"
    write_image(source)
    with pytest.raises(ValueError, match="bounds"):
        Predictor(config).infer(source, "job", box)
    assert calls == []


def test_exact_image_boundary_is_valid(config, monkeypatch):
    calls = []
    patch_model(monkeypatch, calls)
    source = config.paths.image_input / "top.jpg"
    write_image(source)
    assert Predictor(config).infer(source, "job", CropBox(0, 0, 80, 60)).is_ok
    assert calls[0].size == (80, 60)


def test_shared_files_with_different_crops_have_independent_component_results(
        config, service_factory, monkeypatch):
    calls = []
    patch_model(monkeypatch, calls, decide=lambda frame: frame.getpixel((0, 0))[1] > 0)
    source = write_xml(config, pairs=[("top.png", "side.png")] * 3)
    root = ET.parse(source).getroot()
    components = root.findall("./Board/Component")
    for index, component in enumerate(components):
        image = component.find("./CompImage/Image")
        x1, x2 = (40, 80) if index == 1 else (0, 40)
        image.attrib.update(X1=str(x1), Y1="0", X2=str(x2), Y2="60")
    source.write_bytes(ET.tostring(root))
    originals = {}
    for name in ("top.png", "side.png"):
        image = Image.new("RGB", (80, 60), "red")
        image.paste("green", (0, 0, 40, 60))
        path = config.paths.image_input / name
        image.save(path)
        originals[name] = path.read_bytes()
    service = service_factory(Predictor(config))
    service.tick()
    assert service.state.counts() == {"done": 1}
    # Identical ROIs are shared across components; distinct ROIs are never conflated.
    assert len(calls) == 4
    assert all(frame.size == (40, 60) for frame in calls)
    result = ET.parse(config.paths.xml_output / source.name).getroot()
    assert [c.get("AI_result") for c in result.findall("./Board/Component")] == ["OK", "NG", "OK"]
    job = service.state.jobs("done")[0]
    assert len(job["payload"]["manifest"]) == 3  # Archive originals once, not four crops.
    assert len(job["payload"]["images"]) == 4
    assert all(item["coordinate_space"] == "crop" for item in job["payload"]["images"].values())
    assert {item["crop"]["x1"] for item in job["payload"]["images"].values()} == {0, 40}
    done = archive_folder(source.name, config.paths.archive_dir)
    for name in originals:
        assert (done / name).read_bytes() == originals[name]
        with Image.open(done / name) as image:
            assert image.size == (80, 60)


def test_equal_rois_with_signed_zero_share_the_same_verdict_key():
    raw = b'''<Panel><Board><Component Status="FAIL" Type="TEST_PART"><CompImage>
    <Image PicPath1="top.jpg" PicPath2="side.jpg" X1="0" Y1="0" X2="20" Y2="20"/>
    <Image PicPath1="top.jpg" PicPath2="side.jpg" X1="-0.0" Y1="-0" X2="20.0" Y2="20.0"/>
    </CompImage></Component></Board></Panel>'''
    document = Document(raw, ["FAIL"], [".jpg"], {"TEST_PART"})
    assert len(document.regions) == 2
    assert ET.fromstring(document.result({r.key: True for r in document.regions})).find(
        "./Board/Component").get("AI_result") == "OK"


def test_supplied_xml_missing_roi_is_retained_without_inference(config, service_factory, clock):
    fixture = Path(__file__).parent / "fixtures/20261111000316_HA996AQ010070A_T.xml"
    original = fixture.read_bytes()
    config.watch.failure_statuses = ["FAIL"]
    root = ET.fromstring(original)
    config.paths.bom_list.write_text("\n".join({c.get("Type") for c in root.findall("./Board/Component")}))
    source = config.paths.xml_input / fixture.name
    source.write_bytes(original)
    service = service_factory()
    service.tick()
    clock.advance()
    service.tick()
    assert service.state.counts() == {"failed": 1}
    assert "X2" in service.state.jobs("failed")[0]["error"]
    assert service.predictor.calls == []
    assert not list(config.paths.xml_output.iterdir())
    assert not list(config.paths.archive_dir.iterdir())
    assert source.read_bytes() == original


def test_invalid_roi_keeps_xml_and_images(config, service_factory, clock):
    source = write_xml(config)
    root = ET.parse(source).getroot()
    root.find("./Board/Component/CompImage/Image").set("X2", "999")
    source.write_bytes(ET.tostring(root))
    for name in ("top.jpg", "side.jpg"):
        write_image(config.paths.image_input / name)
    service = service_factory()
    service.tick()
    clock.advance()
    service.tick()
    assert service.state.counts() == {"failed": 1}
    assert source.exists()
    assert len(list(config.paths.image_input.iterdir())) == 2
    assert not list(config.paths.xml_output.iterdir())
    assert not list(config.paths.archive_dir.iterdir())


def test_upgrade_reinfers_unpublished_whole_image_results(config, service_factory):
    source = write_xml(config)
    for name in ("top.jpg", "side.jpg"):
        write_image(config.paths.image_input / name)
    first = service_factory()
    xml_hash = digest(source)
    first.state.discover(first.job_id(source, xml_hash), source, xml_hash, first.clock())
    first.process(first.state.jobs("waiting")[0])
    job = first.state.jobs("publishing")[0]
    job["payload"].pop("inference_schema")  # Simulate the previous service version's cache.
    first.state.update(job["id"], first.clock(), payload=job["payload"])
    close_service(first)
    predictor = FakePredictor()
    resumed = service_factory(predictor)
    assert resumed.state.counts() == {"waiting": 1}
    resumed.tick()
    assert predictor.calls == ["top.jpg", "side.jpg"]
    assert len(predictor.crops) == 2
    assert resumed.state.counts() == {"done": 1}
