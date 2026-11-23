from __future__ import annotations

import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from aoi_service.config import Config, load_config
from aoi_service.files import archive_folder, image_basename
from aoi_service.xmlio import Document
from aoi_service.yolo_rules import Detection, ObbRules, classify_detections


@pytest.mark.parametrize("section,values", [
    ("inference", {"conf": 1.1}), ("inference", {"ok_count": 0}),
    ("inference", {"ok_count": True}), ("inference", {"ok_overlap_iou": -1}),
    ("inference", {"center_tolerance": 0.6}), ("inference", {"ok_angle_tolerance": 91}),
    ("inference", {"conf": math.nan}), ("inference", {"unknown_option": 1}),
    ("watch", {"poll_seconds": 0}), ("archive", {"retention_days": 0}),
    ("runtime", {"dry_run": "false"}), ("runtime", {"quiet": True, "verbose": True}),
])
def test_invalid_settings_rejected(section, values):
    with pytest.raises(ValueError):
        Config.model_validate({section: values})


def test_json_relative_paths_and_skip_count(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"inference": {"ok_count": None}}), encoding="utf-8-sig")
    config = load_config(path)
    assert config.paths.xml_input == tmp_path / "input/xml"
    assert config.paths.archive_dir == tmp_path / "input/xml/DONE"
    assert config.paths.bom_list == tmp_path / "bom/1001_T1_BOM_List.txt"
    assert config.watch.failure_statuses == ["False Call"]
    assert config.inference.ok_count is None


@pytest.mark.parametrize("paths", [
    {"xml_input": "same", "image_input": "same"},
    {"xml_output": "input/xml/output"},
    {"archive_dir": "output/subfolder"},
    {"archive_dir": "input"},
])
def test_dangerous_overlapping_paths_rejected(tmp_path, paths):
    with pytest.raises(ValueError):
        Config.model_validate({"paths": paths}).resolve_paths(tmp_path)


@pytest.mark.parametrize("value", ["folder/../a.jpg", r"C:\old\path\a.jpg", "/old/path/a.jpg"])
def test_basename_only(value):
    assert image_basename(value) == "a.jpg"


@pytest.mark.parametrize("value", ["", "..", "dir/", "CON.jpg", "image.jpg:stream", "a.jpg "])
def test_bad_image_names_rejected(value):
    with pytest.raises(ValueError):
        image_basename(value)


@pytest.mark.parametrize("filename", [
    "20261111000316_HA996AQ010070A.xml",
    "20261111000316_BOARD_WITH_UNDERSCORES.xml",
    "20261111000316_HA996AQ010070A.XML",
    "20261111000316_HA996AQ010070A_T.xml",
])
def test_archive_timestamp_parsing(tmp_path, filename):
    assert archive_folder(filename, tmp_path) == (
        tmp_path / "2026-11-11/20261111000316")


@pytest.mark.parametrize("filename", [
    "20260230000316_BOARD.xml", "20261111000316_.xml",
    "20261111000316.xml", "202611110003_BOARD.xml",
])
def test_invalid_archive_filename_rejected(tmp_path, filename):
    with pytest.raises(ValueError):
        archive_folder(filename, tmp_path)


def test_multiple_image_nodes_and_windows_paths():
    raw = b'''<Panel><Board><Component Status="FAIL" Type="TEST_PART" AI_result="">
    <CompImage><Image PicPath1="old/a.jpg" PicPath2="old/b.jpg" X1="1" Y1="2" X2="10" Y2="20"/>
    <Image PicPath1="old\\c.jpg" PicPath2="old\\d.jpg" X1="3" Y1="4" X2="30" Y2="40"/></CompImage>
    <Window PicPath1="ignored.jpg" AI_result="untouched"/></Component>
    <Component Status="PASS" AI_result="keep"/></Board></Panel>'''
    document = Document(raw, ["FAIL"], [".jpg"], {"TEST_PART"})
    assert document.image_names == ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    result = ET.fromstring(document.result({region.key: region.filename != "c.jpg"
                                          for region in document.regions}))
    first, second = result.findall("./Board/Component")
    assert first.get("AI_result") == "NG"
    assert second.get("AI_result") == "keep"
    assert first.find("Window").get("AI_result") == "untouched"


def test_missing_second_light_is_an_error():
    raw = b'<Panel><Board><Component Status="FAIL" Type="TEST_PART"><CompImage><Image PicPath1="a.jpg" X1="0" Y1="0" X2="10" Y2="10"/></CompImage></Component></Board></Panel>'
    with pytest.raises(ValueError):
        Document(raw, ["FAIL"], [".jpg"], {"TEST_PART"})


def rectangle(x1=40, y1=45, x2=60, y2=55, class_id=0, confidence=0.9):
    return Detection(class_id, "OK" if class_id == 0 else "defect", confidence,
                     (x1, y1, x2, y2), ((x1, y1), (x2, y1), (x2, y2), (x1, y2)))


def classify(items, count=None, **kwargs):
    return classify_detections(items, 0, count, (100, 100), 0.25, **kwargs)


def test_original_rules_duplicate_count_and_no_detection():
    ok = rectangle()
    assert classify([ok, ok], count=1).is_ok
    assert not classify([ok, ok], count=2).is_ok
    assert classify([ok, ok], count=2, ok_overlap_iou=0).is_ok
    assert not classify([]).is_ok
    assert not classify([rectangle(class_id=1)]).is_ok


def test_original_rules_contained_ng_and_outside_ng():
    ok = rectangle()
    assert not classify([ok, rectangle(46, 46, 48, 48, class_id=1)]).is_ok
    assert classify([ok, rectangle(0, 0, 5, 5, class_id=1)]).is_ok
    assert not classify([rectangle(0, 0, 10, 5)]).is_ok


def test_obb_rotated_containment_uses_polygon_not_enclosing_box():
    diamond = Detection(0, "OK", 0.9, (30, 30, 70, 70),
                        ((50, 30), (70, 50), (50, 70), (30, 50)))
    assert classify([diamond, rectangle(31, 31, 35, 35, class_id=1)]).is_ok
    assert not classify([diamond, rectangle(48, 48, 52, 52, class_id=1)]).is_ok


def test_obb_angle_center_and_pairwise_gap_limits():
    assert not classify([rectangle(45, 40, 55, 60)], obb_rules=ObbRules(
        angle_target_deg=0, angle_tolerance_deg=5)).is_ok
    assert not classify([rectangle(), rectangle(75, 45, 85, 55)], obb_rules=ObbRules(
        max_center_distance_px=10)).is_ok
    assert not classify([rectangle(), rectangle(65, 45, 75, 55)], obb_rules=ObbRules(
        min_edge_distance_px=6)).is_ok
    assert classify([rectangle()], obb_rules=ObbRules(angle_tolerance_deg=5)).is_ok
