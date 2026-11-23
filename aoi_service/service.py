"""Single inference worker with a durable publish/archive journal."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Protocol

from .config import Config
from .files import ProcessLock, archive_folder, atomic_bytes, digest, direct_file, verified_copy
from .regions import CropBox
from .state import State
from .xmlio import Document, referenced_names
from .yolo_rules import Decision

LOG = logging.getLogger(__name__)
INFERENCE_SCHEMA = "image-roi-v1"


class InferenceBackend(Protocol):
    def infer(self, source: Path, job_id: str, crop: CropBox) -> Decision: ...


class NotReady(Exception):
    pass


class Service:
    def __init__(self, config: Config, predictor: InferenceBackend | Callable[[], InferenceBackend],
                 clock: Callable[[], float] = time.time):
        self.config, self.clock = config, clock
        self.process_lock = ProcessLock(config.paths.state_db.with_suffix(".lock"))
        try:
            self.state = State(config.paths.state_db)
            # Never publish cached whole-image results after upgrading to ROI inference.
            pending = [] if config.runtime.dry_run else self.state.jobs("publishing")
            for job in pending:
                if job["payload"].get("inference_schema") != INFERENCE_SCHEMA:
                    self.state.update(job["id"], self.clock(), status="waiting", attempts=0,
                                      created=self.clock(),
                                      next_try=0, payload={},
                                      error="Unpublished whole-image result invalidated; ROI inference required")
            self.predictor = predictor() if callable(predictor) else predictor
        except BaseException:
            if hasattr(self, "state"):
                self.state.close()
            self.process_lock.close()
            raise
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.observations: dict[Path, tuple[tuple[int, int], float]] = {}
        self.current_job: str | None = None
        self.last_cycle: float | None = None
        self.last_error: str | None = None
        self.next_retention_check = 0.0

    def start(self) -> None:
        self.thread = threading.Thread(target=self.run, name="aoi-worker", daemon=False)
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self.state.close()
        self.process_lock.close()

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.tick()
                self.last_error = None
            except Exception as exc:
                self.last_error = str(exc)
                LOG.exception("Folder scan failed; retrying next cycle")
            finally:
                self.last_cycle = self.clock()
            self.stop_event.wait(self.config.watch.poll_seconds)

    def input_xmls(self) -> list[Path]:
        # Flat input only. DONE and all other subfolders are excluded.
        return sorted(p for p in self.config.paths.xml_input.iterdir()
                      if p.suffix.lower() == ".xml" and p.is_file() and not p.is_symlink())

    def stable(self, path: Path) -> bool:
        try:
            stat = path.stat()
        except FileNotFoundError:
            self.observations.pop(path, None)
            return False
        signature = (stat.st_size, stat.st_mtime_ns)
        previous = self.observations.get(path)
        if previous is None or previous[0] != signature:
            self.observations[path] = (signature, self.clock())
        return self.clock() - self.observations[path][1] >= self.config.watch.stable_seconds

    def job_id(self, path: Path, xml_hash: str) -> str:
        suffix = "dry-run" if self.config.runtime.dry_run else "production"
        return hashlib.sha256(f"{path}\0{xml_hash}\0{suffix}".encode()).hexdigest()

    def tick(self) -> None:
        for source in self.input_xmls():
            if self.stable(source):
                try:
                    direct_file(source, self.config.paths.xml_input)
                    xml_hash = digest(source)
                    self.state.discover(self.job_id(source, xml_hash), source, xml_hash, self.clock())
                except OSError:
                    LOG.warning("XML disappeared or could not be read: %s", source, exc_info=True)
        for job in self.state.jobs("waiting"):
            if self.stop_event.is_set():
                return
            if (job["id"] == self.job_id(Path(job["source"]), job["xml_hash"])
                    and job["next_try"] <= self.clock()):
                self.process(job)
        pending = [] if self.config.runtime.dry_run else self.state.jobs("publishing", "archiving")
        for job in pending:
            if self.stop_event.is_set():
                return
            if job["next_try"] <= self.clock():
                self.finish(job)
        if not self.config.runtime.dry_run and self.clock() >= self.next_retention_check:
            self.expire_archives()
            self.next_retention_check = self.clock() + 3600
        live = {Path(j["source"]) for j in self.state.jobs("waiting", "publishing", "archiving")}
        self.observations = {p: value for p, value in self.observations.items()
                             if p.exists() or p in live}

    def defer(self, job: dict, error: Exception, waiting: bool = False) -> None:
        now = self.clock()
        if waiting:
            failed = now - job["created"] >= self.config.watch.image_wait_timeout_seconds
            status = "failed" if failed else "waiting"
            attempts = job["attempts"]
            delay = self.config.watch.poll_seconds
        else:
            attempts = job["attempts"] + 1
            status = "failed" if attempts >= self.config.watch.max_retries else "waiting"
            delay = self.config.watch.retry_seconds
        self.state.update(job["id"], now, status=status, attempts=attempts,
                          error=str(error), next_try=now + delay)
        if status == "failed":
            LOG.error("Job failed; source retained: %s: %s", job["source"], error)
        elif not waiting:
            LOG.warning("Job will retry: %s: %s", job["source"], error)

    def process(self, job: dict) -> None:
        self.current_job = job["source"]
        try:
            source = Path(job["source"])
            if not source.exists():
                self.state.update(job["id"], self.clock(), status="superseded",
                                  error="Source no longer exists")
                return
            direct_file(source, self.config.paths.xml_input)
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != job["xml_hash"]:
                self.state.update(job["id"], self.clock(), status="superseded",
                                  error="Source XML was replaced before processing")
                return
            if not self.stable(source):
                raise NotReady("XML is still being written")
            destination = archive_folder(source.name, self.config.paths.archive_dir)
            try:
                document = Document(data, self.config.watch.failure_statuses,
                                    self.config.inference.ext)
            except ET.ParseError as exc:
                raise NotReady(f"XML is incomplete or malformed: {exc}") from exc
            paths = [self.config.paths.image_input / name for name in document.image_names]
            for path in paths:
                direct_file(path, self.config.paths.image_input)
            unsettled = [path.name for path in paths if not path.is_file() or not self.stable(path)]
            if unsettled:
                raise NotReady("Waiting for complete images: " + ", ".join(unsettled))
            manifest = [{"source": str(p), "sha256": digest(p), "name": p.name,
                         "kind": "xml" if p == source else "image"}
                        for p in [source, *paths]]
            if manifest[0]["sha256"] != job["xml_hash"]:
                raise NotReady("XML changed during readiness checks")
            self.state.update(job["id"], self.clock(), status="processing")
            decisions: dict[str, Decision] = {}
            # No early exit after an NG verdict: both light sources must run.
            for region in document.regions:
                path = self.config.paths.image_input / region.filename
                decisions[region.key] = self.predictor.infer(path, job["id"], region.crop)
                decision = decisions[region.key]
                LOG.info("%s / %s crop=%s -> %s: %s", source.name, path.name,
                         region.crop.as_tuple(), "OK" if decision.is_ok else "NG", decision.reason)
            for item in manifest:
                if digest(Path(item["source"])) != item["sha256"]:
                    raise RuntimeError(f"Input changed during inference: {item['source']}")
            result = document.result({name: d.is_ok for name, d in decisions.items()})
            ET.fromstring(result)
            payload = {
                "inference_schema": INFERENCE_SCHEMA,
                "manifest": manifest, "archive_dir": str(destination),
                "after_success": self.config.archive.after_success,
                "archive_copied": False, "output": str(self.config.paths.xml_output / source.name),
                "output_hash": hashlib.sha256(result).hexdigest(),
                "result_xml": result.decode("utf-8"), "component_count": len(document.components),
                "images": {
                    region.key: {
                        "source_image": region.filename, "crop": asdict(region.crop),
                        "coordinate_space": "crop",
                        "verdict": "OK" if decisions[region.key].is_ok else "NG",
                        "reason": decisions[region.key].reason,
                        "measurements": asdict(decisions[region.key].measurements)
                        if decisions[region.key].measurements else None,
                    } for region in document.regions
                },
            }
            status = "dry_run" if self.config.runtime.dry_run else "publishing"
            self.state.update(job["id"], self.clock(), status=status, payload=payload,
                              error=None, next_try=0)
            LOG.info("Inference complete: %s (%d components, %d image regions)%s", source.name,
                     len(document.components), len(decisions), " [dry run]" if status == "dry_run" else "")
        except NotReady as exc:
            self.defer(job, exc, waiting=True)
        except Exception as exc:
            self.defer(job, exc)
            LOG.debug("Processing exception", exc_info=True)
        finally:
            self.current_job = None

    def finish(self, job: dict) -> None:
        try:
            if job["status"] == "publishing":
                self.publish(job)
                job = self.state.get(job["id"])
            self.finalize(job)
        except Exception as exc:
            # Never return a committed job to inference. Retry IO without a limit.
            self.state.update(job["id"], self.clock(), error=str(exc),
                              next_try=self.clock() + self.config.watch.retry_seconds)
            LOG.warning("Output/archive pending for %s: %s", job["source"], exc)

    def publish(self, job: dict) -> None:
        payload = job["payload"]
        output = Path(payload["output"])
        if output.is_symlink():
            raise ValueError(f"Output must not be a symlink: {output}")
        if output.exists():
            if digest(output) != payload["output_hash"]:
                raise FileExistsError(f"Different result already exists: {output}; move it before retrying")
        else:
            # On recovery, do not publish an uncommitted result for changed inputs.
            for item in payload["manifest"]:
                if digest(Path(item["source"])) != item["sha256"]:
                    raise RuntimeError(f"Input changed before output publication: {item['source']}")
            atomic_bytes(output, payload["result_xml"].encode("utf-8"))
        if digest(output) != payload["output_hash"]:
            raise RuntimeError(f"Output integrity verification failed: {output}")
        ET.fromstring(output.read_bytes())
        payload["published_at"] = self.clock()
        self.state.update(job["id"], self.clock(), status="archiving", payload=payload,
                          error=None, next_try=0)
        LOG.info("Published result XML: %s", output)

    def protected_images(self, current: dict) -> set[str] | None:
        """Fresh scan protects unregistered XMLs too; malformed XML defers cleanup."""
        protected: set[str] = set()
        for source in self.input_xmls():
            try:
                raw = source.read_bytes()
                xml_hash = hashlib.sha256(raw).hexdigest()
                key = self.job_id(source, xml_hash)
                if key == current["id"]:
                    continue
                known = self.state.find(key)
                if known is not None:
                    payload = known["payload"]
                    if (known["status"] == "done" or
                            (known["status"] == "archiving" and
                             (payload.get("archive_copied") or payload.get("after_success") == "delete"))):
                        continue
                protected.update(name.casefold() for name in referenced_names(raw))
            except (OSError, ET.ParseError, ValueError):
                return None
        return protected

    def finalize(self, job: dict) -> None:
        payload = job["payload"]
        mode = payload["after_success"]
        if mode == "keep":
            self.state.update(job["id"], self.clock(), status="done", error=None)
            return
        source_xml = Path(job["source"])
        if source_xml.exists() and digest(source_xml) == job["xml_hash"]:
            for item in payload["manifest"]:
                source = Path(item["source"])
                if (item["kind"] == "image" and source.exists()
                        and digest(source) != item["sha256"]):
                    raise RuntimeError(f"Image changed after inference; retaining source XML: {source}")
        if mode == "archive":
            directory = Path(payload["archive_dir"])
            root = self.config.paths.archive_dir.resolve()
            if not directory.resolve().is_relative_to(root):
                raise ValueError("Archive directory is outside configured archive root")
            for item in payload["manifest"]:
                source, target = Path(item["source"]), directory / item["name"]
                verified_copy(source, target, item["sha256"])
            if not payload["archive_copied"]:
                payload["archive_copied"] = True
                self.state.update(job["id"], self.clock(), payload=payload)
        protected = self.protected_images(job)
        deferred = False
        for item in payload["manifest"]:
            source = Path(item["source"])
            if item["kind"] == "image" and (protected is None or item["name"].casefold() in protected):
                deferred = deferred or source.exists()
                continue
            if source.exists():
                root = self.config.paths.xml_input if item["kind"] == "xml" else self.config.paths.image_input
                direct_file(source, root)
                # A newly arrived file reusing a name belongs to a different job.
                if digest(source) == item["sha256"]:
                    source.unlink()
        if deferred:
            self.state.update(job["id"], self.clock(), error="Waiting for other XML references before source cleanup",
                              next_try=self.clock() + self.config.watch.poll_seconds)
            return
        payload["completed_at"] = self.clock()
        self.state.update(job["id"], self.clock(), status="done", payload=payload, error=None)
        LOG.info("Job completed (%s): %s", mode, job["source"])

    def expire_archives(self) -> None:
        days = self.config.archive.retention_days
        if days is None:
            return
        cutoff = self.clock() - days * 86400
        jobs = self.state.jobs()
        protected = {
            str(Path(j["payload"]["archive_dir"]) / item["name"])
            for j in jobs if j["status"] in ("publishing", "archiving")
            for item in j["payload"].get("manifest", [])
            if j["payload"].get("after_success") == "archive"
        }
        # Newer completed jobs can share archive files with older jobs.
        protected.update(
            str(Path(j["payload"]["archive_dir"]) / item["name"])
            for j in jobs if j["status"] == "done" and j["payload"].get("completed_at", j["updated"]) > cutoff
            for item in j["payload"].get("manifest", [])
            if j["payload"].get("after_success") == "archive"
        )
        for job in jobs:
            payload = job["payload"]
            if (job["status"] != "done" or payload.get("after_success") != "archive"
                    or payload.get("archive_purged")
                    or payload.get("completed_at", job["updated"]) > cutoff):
                continue
            directory = Path(payload["archive_dir"])
            if not directory.resolve().is_relative_to(self.config.paths.archive_dir.resolve()):
                continue
            deferred = False
            try:
                for item in payload["manifest"]:
                    target = directory / item["name"]
                    if str(target) in protected:
                        deferred = True
                    elif target.exists() and not target.is_symlink():
                        if digest(target) != item["sha256"]:
                            raise RuntimeError(f"Archive changed; refusing retention cleanup: {target}")
                        target.unlink()
                if not deferred:
                    payload["archive_purged"] = True
                    self.state.update(job["id"], self.clock(), payload=payload)
                for folder in (directory, directory.parent):
                    if folder.exists() and not any(folder.iterdir()):
                        folder.rmdir()
            except (OSError, RuntimeError):
                LOG.warning("Retention cleanup failed: %s", directory, exc_info=True)

    def status(self) -> dict[str, Any]:
        return {"worker_alive": self.thread is not None and self.thread.is_alive(),
                "current_xml": self.current_job, "last_cycle": self.last_cycle,
                "last_error": self.last_error, "dry_run": self.config.runtime.dry_run,
                "jobs": self.state.counts()}
