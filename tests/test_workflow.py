from __future__ import annotations

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from aoi_service.files import archive_folder, digest
from aoi_service.xmlio import Document
from conftest import FakePredictor, close_service, write_image, write_xml


@pytest.mark.parametrize("first,second,expected", [
    (True, True, "OK"), (False, True, "NG"),
    (True, False, "NG"), (False, False, "NG"),
])
def test_both_lights_and_archive(config, service_factory, first, second, expected):
    source = write_xml(config)
    original = source.read_bytes()
    for name in ("top.jpg", "side.jpg"):
        write_image(config.paths.image_input / name)
    predictor = FakePredictor({"top.jpg": first, "side.jpg": second})
    service = service_factory(predictor)
    service.tick()
    assert predictor.calls == ["top.jpg", "side.jpg"]
    output = config.paths.xml_output / source.name
    root = ET.parse(output).getroot()
    comp = root.find("./Board/Component")
    assert comp.get("AI_result") == expected
    assert root.get("Status") == "FAIL"
    assert comp.find("Window").get("AI_result") == "unchanged"
    done = config.paths.archive_dir / "2026-11-11" / "20261111000316"
    assert {p.name for p in done.iterdir()} == {source.name, "top.jpg", "side.jpg"}
    assert (done / source.name).read_bytes() == original
    assert not source.exists()
    assert not list(config.paths.image_input.iterdir())
    assert service.state.counts() == {"done": 1}
    service.tick()
    assert len(predictor.calls) == 2  # DONE is never rescanned.


def test_sample_structure_17_components_34_crops_with_complete_coordinates(config, service_factory):
    fixture = Path(__file__).parent / "fixtures/20261111000316_HA996AQ010070A_T.xml"
    # Keep the supplied fixture unchanged. It has empty X2/Y2; populate test-only pixel ROIs.
    root = ET.fromstring(fixture.read_bytes())
    for image in root.findall("./Board/Component/CompImage/Image"):
        image.attrib.update(X1="10", Y1="10", X2="70", Y2="50")
    data = ET.tostring(root)
    source = config.paths.xml_input / fixture.name.replace("_T.xml", ".xml")
    source.write_bytes(data)
    document = Document(data, ["FAIL"], config.inference.ext)
    assert len(document.components) == 17
    assert len(document.image_names) == 34
    for name in document.image_names:
        write_image(config.paths.image_input / name)
    predictor = FakePredictor({document.image_names[1]: False})
    service = service_factory(predictor)
    service.tick()
    result = ET.parse(config.paths.xml_output / source.name).getroot()
    comps = result.findall("./Board/Component")
    assert [c.get("AI_result") for c in comps] == ["NG"] + ["OK"] * 16
    original = ET.fromstring(data)
    for before, after in zip(original.iter(), result.iter(), strict=True):
        attrs = dict(after.attrib)
        if before.tag == "Component":
            attrs["AI_result"] = before.get("AI_result")
        assert before.tag == after.tag
        assert before.attrib == attrs
        assert before.text == after.text
    assert len(predictor.calls) == 34
    done = archive_folder(source.name, config.paths.archive_dir)
    assert len(list(done.iterdir())) == 35


def test_missing_images_do_not_block_another_xml(config, service_factory, clock):
    missing = write_xml(config, "20261111000315_A.xml", [("missing.jpg", "late.jpg")])
    ready = write_xml(config, "20261111000316_B.xml")
    for name in ("top.jpg", "side.jpg"):
        write_image(config.paths.image_input / name)
    service = service_factory()
    service.tick()
    assert missing.exists()
    assert not ready.exists()
    assert service.state.counts() == {"done": 1, "waiting": 1}
    clock.advance(31)
    service.tick()
    assert service.state.counts() == {"done": 1, "failed": 1}
    for name in ("missing.jpg", "late.jpg"):
        write_image(config.paths.image_input / name)
    service.state.retry_failed(clock())
    service.tick()
    assert service.state.counts() == {"done": 2}


def test_file_stability_and_late_arrival(config, service_factory, clock):
    config.watch.stable_seconds = 2
    source = write_xml(config)
    predictor = FakePredictor()
    service = service_factory(predictor)
    service.tick()
    assert service.state.counts() == {}
    clock.advance(2)
    service.tick()
    assert service.state.counts() == {"waiting": 1}
    for name in ("top.jpg", "side.jpg"):
        write_image(config.paths.image_input / name)
    clock.advance(1)
    service.tick()
    assert not predictor.calls
    clock.advance(2)
    service.tick()
    assert service.state.counts() == {"done": 1}
    assert not source.exists()


def test_inference_error_keeps_all_sources(config, service_factory, clock):
    source = write_xml(config)
    for name in ("top.jpg", "side.jpg"):
        write_image(config.paths.image_input / name)
    service = service_factory(FakePredictor(error=RuntimeError("GPU failure")))
    service.tick()
    clock.advance()
    service.tick()
    assert service.state.counts() == {"failed": 1}
    assert source.exists()
    assert len(list(config.paths.image_input.iterdir())) == 2
    assert not list(config.paths.xml_output.iterdir())


def test_restart_after_inference_before_publication(config, service_factory):
    source = write_xml(config)
    for name in ("top.jpg", "side.jpg"):
        write_image(config.paths.image_input / name)
    first = service_factory()
    xml_hash = digest(source)
    first.state.discover(first.job_id(source, xml_hash), source, xml_hash, first.clock())
    first.process(first.state.jobs("waiting")[0])
    assert first.state.counts() == {"publishing": 1}
    assert not list(config.paths.xml_output.iterdir())
    close_service(first)
    predictor = FakePredictor()
    resumed = service_factory(predictor)
    resumed.tick()
    assert resumed.state.counts() == {"done": 1}
    assert predictor.calls == []


def test_partial_archive_retry_never_reinfers(config, service_factory, clock, monkeypatch):
    import aoi_service.service as module
    source = write_xml(config)
    for name in ("top.jpg", "side.jpg"):
        write_image(config.paths.image_input / name)
    original_copy = module.verified_copy
    counter = 0

    def interrupted_copy(*args):
        nonlocal counter
        counter += 1
        if counter == 2:
            raise OSError("Simulated network interruption")
        return original_copy(*args)

    monkeypatch.setattr(module, "verified_copy", interrupted_copy)
    first = service_factory()
    first.tick()
    assert first.state.counts() == {"archiving": 1}
    assert (config.paths.xml_output / source.name).exists()
    assert source.exists()
    close_service(first)
    monkeypatch.setattr(module, "verified_copy", original_copy)
    predictor = FakePredictor()
    resumed = service_factory(predictor)
    clock.advance()
    resumed.tick()
    assert resumed.state.counts() == {"done": 1}
    assert predictor.calls == []


def test_output_conflict_blocks_cleanup_and_can_resume(config, service_factory, clock):
    source = write_xml(config)
    for name in ("top.jpg", "side.jpg"):
        write_image(config.paths.image_input / name)
    output = config.paths.xml_output / source.name
    output.write_bytes(b"existing unrelated output")
    predictor = FakePredictor()
    service = service_factory(predictor)
    service.tick()
    assert service.state.counts() == {"publishing": 1}
    assert output.read_bytes() == b"existing unrelated output"
    assert source.exists()
    output.unlink()
    clock.advance()
    service.tick()
    assert service.state.counts() == {"done": 1}
    assert len(predictor.calls) == 2


def test_shared_image_is_not_removed_before_other_xml_finishes(config, service_factory, clock):
    first = write_xml(config, "20261111000316_A.xml", [("shared.jpg", "first.jpg")])
    second = write_xml(config, "20261111000317_B.xml", [("shared.jpg", "late.jpg")])
    for name in ("shared.jpg", "first.jpg"):
        write_image(config.paths.image_input / name)
    service = service_factory()
    service.tick()
    assert not first.exists()
    assert second.exists()
    assert (config.paths.image_input / "shared.jpg").exists()
    assert service.state.counts() == {"archiving": 1, "waiting": 1}
    write_image(config.paths.image_input / "late.jpg")
    clock.advance()
    service.tick()
    clock.advance()
    service.tick()
    assert service.state.counts() == {"done": 2}
    assert not list(config.paths.image_input.iterdir())
    for stamp in ("20261111000316", "20261111000317"):
        assert (config.paths.archive_dir / "2026-11-11" / stamp / "shared.jpg").exists()


def test_dry_run_has_no_output_or_cleanup_and_does_not_consume_real_job(config, service_factory):
    source = write_xml(config)
    for name in ("top.jpg", "side.jpg"):
        write_image(config.paths.image_input / name)
    config.runtime.dry_run = True
    first = service_factory()
    first.tick()
    assert first.state.counts() == {"dry_run": 1}
    assert source.exists()
    assert not list(config.paths.xml_output.iterdir())
    assert not list(config.paths.archive_dir.iterdir())
    close_service(first)
    config.runtime.dry_run = False
    real = service_factory()
    real.tick()
    assert real.state.counts() == {"dry_run": 1, "done": 1}


def test_archive_collision_keeps_sources(config, service_factory):
    source = write_xml(config)
    for name in ("top.jpg", "side.jpg"):
        write_image(config.paths.image_input / name)
    done = archive_folder(source.name, config.paths.archive_dir)
    done.mkdir(parents=True)
    (done / "top.jpg").write_bytes(b"different image")
    service = service_factory()
    service.tick()
    assert service.state.counts() == {"archiving": 1}
    assert source.exists()
    assert (done / "top.jpg").read_bytes() == b"different image"


def test_retention_deletes_only_journaled_archive_files(config, service_factory, clock):
    config.archive.retention_days = 1
    source = write_xml(config)
    for name in ("top.jpg", "side.jpg"):
        write_image(config.paths.image_input / name)
    service = service_factory()
    service.tick()
    done = archive_folder(source.name, config.paths.archive_dir)
    extra = done / "user-note.txt"
    extra.write_text("keep")
    clock.advance(86401)
    service.tick()
    assert list(done.iterdir()) == [extra]
    assert (config.paths.xml_output / source.name).exists()


def test_input_changed_during_inference_is_not_published(config, service_factory):
    source = write_xml(config)
    for name in ("top.jpg", "side.jpg"):
        write_image(config.paths.image_input / name)

    class MutatingPredictor(FakePredictor):
        def infer(self, path, job_id, crop):
            result = super().infer(path, job_id, crop)
            if path.name == "side.jpg":
                (config.paths.image_input / "top.jpg").write_bytes(b"new image contents")
            return result

    service = service_factory(MutatingPredictor())
    service.tick()
    assert source.exists()
    assert not list(config.paths.xml_output.iterdir())
    assert service.state.counts() == {"waiting": 1}
