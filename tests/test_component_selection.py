from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from PIL import Image

from aoi_service.bom import load_bom
from aoi_service.files import archive_folder, digest
from aoi_service.service import Service
from aoi_service.xmlio import Document
from conftest import FakePredictor, close_service, write_image, write_xml


ROOT = Path(__file__).resolve().parents[1]
ONLINE_XML = ROOT / "tests/fixtures/20260407133353_20260407 133353T1.xml"
BOM = ROOT / "bom/1001_T1_BOM_List.txt"


def copy_bom(config):
    config.paths.bom_list.write_bytes(BOM.read_bytes())
    return load_bom(config.paths.bom_list)


@pytest.mark.parametrize("status,part_type,part_no,selected", [
    ("False Call", "TEST_PART", "ignored", True),
    ("false call", " TEST_PART ", "ignored", True),
    ("Fail", "TEST_PART", "ignored", False),
    ("NG", "TEST_PART", "ignored", False),
    ("PASS", "TEST_PART", "ignored", False),
    ("False Call", "OTHER", "TEST_PART", False),
    ("False Call", "TEST_PART-extra", "ignored", False),
    ("False Call", "TEST", "ignored", False),
    ("False Call", "test_part", "ignored", False),
    ("False Call", "", "TEST_PART", False),
])
def test_component_status_and_exact_type_both_required(status, part_type, part_no, selected):
    root = ET.Element("Panel", Status="False Call")
    board = ET.SubElement(root, "Board", Status="False Call")
    comp = ET.SubElement(board, "Component", Status=status, Type=part_type, PartNo=part_no,
                         AI_result="existing-value")
    image_group = ET.SubElement(comp, "CompImage")
    for light in ("solder", "uniform"):
        ET.SubElement(image_group, "Image", PicPath=f"old/{light}.jpg",
                      X1="0", Y1="0", X2="10", Y2="10")
    doc = Document(ET.tostring(root), ["False Call"], [".jpg"], {"TEST_PART"})
    assert len(doc.components) == int(selected)
    assert len(doc.regions) == (2 if selected else 0)
    updated = ET.fromstring(doc.result({region.key: True for region in doc.regions}))
    assert updated.find("./Board/Component").get("AI_result") == ("OK" if selected else "existing-value")


def test_bom_utf8_bom_crlf_whitespace_and_duplicates(tmp_path):
    path = tmp_path / "parts.txt"
    path.write_text(" TEST_PART \r\n\r\nOTHER_PART\r\nTEST_PART\r\n", encoding="utf-8-sig")
    assert load_bom(path) == frozenset({"TEST_PART", "OTHER_PART"})


@pytest.mark.parametrize("contents", ["", "\r\n \r\n", "\ufeff"])
def test_empty_bom_is_a_configuration_error(tmp_path, contents):
    path = tmp_path / "empty.txt"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="no part numbers"):
        load_bom(path)


def test_missing_bom_prevents_startup_before_model_loading(config):
    config.paths.bom_list.unlink()
    loaded = []
    with pytest.raises(ValueError, match="Cannot read BOM"):
        config.prepare(require_model=False)
    with pytest.raises(ValueError, match="Cannot read BOM"):
        Service(config, lambda: loaded.append(True))
    assert loaded == []


def test_actual_online_xml_type_is_not_in_provided_bom(config, service_factory):
    parts = copy_bom(config)
    assert len(parts) == 62
    original = ONLINE_XML.read_bytes()
    source = config.paths.xml_input / ONLINE_XML.name
    source.write_bytes(original)
    service = service_factory()
    service.tick()  # No source images are required for a component outside the BOM.
    assert service.state.counts() == {"done": 1}
    assert service.predictor.calls == []
    output = ET.parse(config.paths.xml_output / source.name).getroot()
    assert ET.tostring(output) == ET.tostring(ET.fromstring(original))
    component = output.find("./Board/Component")
    assert component.get("Type") == "RC0402_APL"
    assert component.get("AI_result") == ""
    job = service.state.jobs("done")[0]
    assert job["payload"]["component_selection"] == {
        "total": 1, "selected": 0, "skipped_status": 0, "skipped_bom": 1,
    }
    done = archive_folder(source.name, config.paths.archive_dir)
    assert done == config.paths.archive_dir / "2026-04-07/20260407133353"
    assert {p.name for p in done.iterdir()} == {source.name}
    assert (done / source.name).read_bytes() == original


@pytest.mark.parametrize("first,second,expected", [
    (True, True, "OK"), (False, True, "NG"),
    (True, False, "NG"), (False, False, "NG"),
])
def test_online_picpath_nodes_crop_both_lights_and_only_modify_selected_ai_result(
        config, service_factory, first, second, expected):
    parts = copy_bom(config)
    root = ET.fromstring(ONLINE_XML.read_bytes())
    comp = root.find("./Board/Component")
    comp.set("Type", sorted(parts)[0])  # Test-only copy to exercise a BOM match.
    # Component status governs inference regardless of Panel/Board status.
    root.set("Status", "PASS")
    root.find("Board").set("Status", "PASS")
    original = ET.tostring(root)
    source = config.paths.xml_input / ONLINE_XML.name
    source.write_bytes(original)
    names = ["R478_1_SolderLight.jpg", "R478_1_UniformLight.jpg"]
    for name in names:
        Image.new("RGB", (640, 480), "white").save(config.paths.image_input / name)
    predictor = FakePredictor(dict(zip(names, (first, second), strict=True)))
    service = service_factory(predictor)
    service.tick()
    assert service.state.counts() == {"done": 1}
    assert predictor.calls == names  # PicPathStd golden images are never read.
    assert [crop.as_tuple() for crop in predictor.crops] == [(165, 175, 475, 305)] * 2
    updated = ET.parse(config.paths.xml_output / source.name).getroot()
    for before, after in zip(root.iter(), updated.iter(), strict=True):
        attributes = dict(after.attrib)
        if before.tag == "Component":
            assert attributes.pop("AI_result") == expected
            original_attributes = {key: value for key, value in before.attrib.items() if key != "AI_result"}
        else:
            original_attributes = before.attrib
        assert attributes == original_attributes
    done = archive_folder(source.name, config.paths.archive_dir)
    assert {p.name for p in done.iterdir()} == {source.name, *names}
    assert (done / source.name).read_bytes() == original


def test_skipped_components_do_not_validate_wait_for_or_move_their_images(config, service_factory):
    source = write_xml(config)
    root = ET.parse(source).getroot()
    board = root.find("Board")
    for status, part_type, name in (("False Call", "OTHER", "not-chiprc"),
                                     ("Fail", "TEST_PART", "machine-fail")):
        comp = ET.SubElement(board, "Component", Status=status, Type=part_type,
                             AI_result="keep-existing", CompName=name)
        group = ET.SubElement(comp, "CompImage")
        # Invalid/missing crop and second light must be ignored for skipped components.
        ET.SubElement(group, "Image", PicPath=f"old/{name}.jpg", X2="")
        (config.paths.image_input / f"{name}.jpg").write_bytes(b"untouched image bytes")
    source.write_bytes(ET.tostring(root))
    for name in ("top.jpg", "side.jpg"):
        write_image(config.paths.image_input / name)
    service = service_factory()
    service.tick()
    assert service.state.counts() == {"done": 1}
    assert service.predictor.calls == ["top.jpg", "side.jpg"]
    result = ET.parse(config.paths.xml_output / source.name).getroot()
    assert [c.get("AI_result") for c in result.findall("./Board/Component")] == [
        "OK", "keep-existing", "keep-existing",
    ]
    assert {p.name for p in config.paths.image_input.iterdir()} == {"not-chiprc.jpg", "machine-fail.jpg"}
    for path in config.paths.image_input.iterdir():
        assert path.read_bytes() == b"untouched image bytes"
    job = service.state.jobs("done")[0]
    assert job["payload"]["component_selection"] == {
        "total": 3, "selected": 1, "skipped_status": 1, "skipped_bom": 1,
    }


def split_picpaths(source):
    root = ET.parse(source).getroot()
    for group in root.findall("./Board/Component/CompImage"):
        for paired in list(group):
            for attr in ("PicPath1", "PicPath2"):
                attributes = {key: value for key, value in paired.attrib.items()
                              if key not in ("PicPath1", "PicPath2")}
                ET.SubElement(group, "Image", PicPath=paired.get(attr), **attributes)
            group.remove(paired)
    source.write_bytes(ET.tostring(root))


def test_online_picpath_shared_source_is_kept_until_other_xml_finishes(config, service_factory, clock):
    first = write_xml(config, "20260407133353_A.xml", [("shared.jpg", "first.jpg")])
    second = write_xml(config, "20260407133354_B.xml", [("shared.jpg", "late.jpg")])
    split_picpaths(first)
    split_picpaths(second)
    for name in ("shared.jpg", "first.jpg"):
        write_image(config.paths.image_input / name)
    service = service_factory()
    service.tick()
    assert service.state.counts() == {"archiving": 1, "waiting": 1}
    assert (config.paths.image_input / "shared.jpg").exists()
    write_image(config.paths.image_input / "late.jpg")
    clock.advance()
    service.tick()
    clock.advance()
    service.tick()
    assert service.state.counts() == {"done": 2}
    assert not list(config.paths.image_input.iterdir())


def test_selected_online_component_requires_both_lights():
    root = ET.fromstring(ONLINE_XML.read_bytes())
    comp = root.find("./Board/Component")
    group = comp.find("CompImage")
    group.remove(group[-1])
    with pytest.raises(ValueError, match="at least two"):
        Document(ET.tostring(root), ["False Call"], [".jpg"], {comp.get("Type")})


def test_changed_bom_invalidates_unpublished_results_and_keeps_skipped_images(config, service_factory):
    source = write_xml(config)
    for name in ("top.jpg", "side.jpg"):
        write_image(config.paths.image_input / name)
    first = service_factory()
    xml_hash = digest(source)
    first.state.discover(first.job_id(source, xml_hash), source, xml_hash, first.clock())
    first.process(first.state.jobs("waiting")[0])
    assert first.state.counts() == {"publishing": 1}
    close_service(first)
    config.paths.bom_list.write_text("OTHER_PART\n", encoding="utf-8")
    predictor = FakePredictor()
    resumed = service_factory(predictor)
    assert resumed.state.counts() == {"waiting": 1}
    resumed.tick()
    assert resumed.state.counts() == {"done": 1}
    assert predictor.calls == []
    assert ET.parse(config.paths.xml_output / source.name).find("./Board/Component").get("AI_result") == ""
    assert len(list(config.paths.image_input.iterdir())) == 2
    assert len(resumed.state.jobs("done")[0]["payload"]["manifest"]) == 1
