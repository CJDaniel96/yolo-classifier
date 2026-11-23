from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from aoi_service.files import archive_folder, digest, verified_copy
from conftest import FakePredictor, close_service, write_image, write_xml


def ready_job(config):
    source = write_xml(config)
    for name in ("top.jpg", "side.jpg"):
        write_image(config.paths.image_input / name)
    return source


def test_dry_run_does_not_resume_pending_production_publication(config, service_factory):
    source = ready_job(config)
    original = service_factory()
    xml_hash = digest(source)
    original.state.discover(original.job_id(source, xml_hash), source, xml_hash, original.clock())
    original.process(original.state.jobs("waiting")[0])
    close_service(original)
    config.runtime.dry_run = True
    preview = service_factory()
    preview.tick()
    assert preview.state.counts() == {"publishing": 1, "dry_run": 1}
    assert source.exists()
    assert not list(config.paths.xml_output.iterdir())


def test_publication_recovery_accepts_identical_existing_output(config, service_factory):
    source = ready_job(config)
    first = service_factory()
    xml_hash = digest(source)
    first.state.discover(first.job_id(source, xml_hash), source, xml_hash, first.clock())
    first.process(first.state.jobs("waiting")[0])
    job = first.state.jobs("publishing")[0]
    output = Path(job["payload"]["output"])
    output.write_text(job["payload"]["result_xml"], encoding="utf-8")
    close_service(first)  # Simulate a crash after rename but before SQLite commit.
    resumed = service_factory(FakePredictor())
    resumed.tick()
    assert resumed.state.counts() == {"done": 1}
    assert resumed.predictor.calls == []
    assert not source.exists()


@pytest.mark.parametrize("mode", ["delete", "keep"])
def test_optional_success_modes(config, service_factory, mode):
    source = ready_job(config)
    config.archive.after_success = mode
    service = service_factory()
    service.tick()
    assert service.state.counts() == {"done": 1}
    assert source.exists() == (mode == "keep")
    assert bool(list(config.paths.image_input.iterdir())) == (mode == "keep")
    assert not list(config.paths.archive_dir.iterdir())


def test_corrupt_image_never_publishes(config, service_factory, clock):
    source = ready_job(config)
    (config.paths.image_input / "top.jpg").write_bytes(b"incomplete JPEG")
    service = service_factory()
    service.tick()
    clock.advance()
    service.tick()
    assert service.state.counts() == {"failed": 1}
    assert source.exists()
    assert not list(config.paths.xml_output.iterdir())


def test_incomplete_xml_times_out_without_source_cleanup(config, service_factory, clock):
    source = config.paths.xml_input / "20261111000316_BOARD.xml"
    source.write_bytes(b"<Panel><Board")
    service = service_factory()
    service.tick()
    assert service.state.counts() == {"waiting": 1}
    clock.advance(31)
    service.tick()
    assert service.state.counts() == {"failed": 1}
    assert source.exists()


def test_unsafe_image_symlink_is_not_inferred_or_deleted(config, service_factory, tmp_path, clock):
    source = ready_job(config)
    victim = tmp_path / "elsewhere.jpg"
    write_image(victim)
    top = config.paths.image_input / "top.jpg"
    top.unlink()
    top.symlink_to(victim)
    service = service_factory()
    service.tick()
    clock.advance()
    service.tick()
    assert source.exists() and victim.exists()
    assert service.predictor.calls == []


def test_empty_fail_list_xml_is_output_without_model_calls(config, service_factory):
    source = config.paths.xml_input / "20261111000316_PASS.xml"
    source.write_bytes(b'<Panel Status="PASS"><Board Status="PASS"/></Panel>')
    service = service_factory()
    service.tick()
    assert service.state.counts() == {"done": 1}
    assert not service.predictor.calls
    assert ET.parse(config.paths.xml_output / source.name).getroot().get("Status") == "PASS"


def test_verified_copy_recovers_when_source_was_already_removed(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"original bytes")
    expected = digest(source)
    verified_copy(source, target, expected)
    source.unlink()
    verified_copy(source, target, expected)
    assert target.read_bytes() == b"original bytes"


def test_image_replaced_after_publication_preserves_source_xml(config, service_factory):
    source = ready_job(config)
    service = service_factory()
    xml_hash = digest(source)
    service.state.discover(service.job_id(source, xml_hash), source, xml_hash, service.clock())
    service.process(service.state.jobs("waiting")[0])
    job = service.state.jobs("publishing")[0]
    service.publish(job)
    # A previous archive attempt copied the old bytes before being interrupted.
    job = service.state.jobs("archiving")[0]
    for item in job["payload"]["manifest"]:
        verified_copy(Path(item["source"]), Path(job["payload"]["archive_dir"]) / item["name"],
                      item["sha256"])
    (config.paths.image_input / "top.jpg").write_bytes(b"replacement image")
    service.finish(job)
    assert source.exists()
    assert service.state.counts() == {"archiving": 1}


def test_retention_does_not_purge_shared_file_used_by_a_recent_job(config, service_factory, clock):
    config.archive.retention_days = 1
    first = write_xml(config, "20261111000316_A.xml")
    for name in ("top.jpg", "side.jpg"):
        write_image(config.paths.image_input / name)
    service = service_factory()
    service.tick()
    done = archive_folder(first.name, config.paths.archive_dir)
    clock.advance(43200)
    second = write_xml(config, "20261111000316_B.xml")
    for name in ("top.jpg", "side.jpg"):
        write_image(config.paths.image_input / name)
    service.tick()
    clock.advance(43201)
    service.tick()
    assert not (done / first.name).exists()
    assert (done / second.name).exists()
    assert (done / "top.jpg").exists()
    assert (done / "side.jpg").exists()
