"""Read AOI XML and update only Component.AI_result."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Set

from .files import image_basename
from .regions import CropBox, ImageRegion


@dataclass
class Component:
    node: ET.Element
    images: list[ImageRegion]


class Document:
    def __init__(self, data: bytes, statuses: list[str], extensions: list[str],
                 allowed_types: Set[str]):
        if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
            raise ValueError("DTD/entity declarations are not supported")
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True, insert_pis=True))
        self.root = ET.fromstring(data, parser=parser)
        if self.root.tag.rsplit("}", 1)[-1] != "Panel":
            raise ValueError("Expected an AOI Panel XML document")
        self.components: list[Component] = []
        self.selection = {"total": 0, "selected": 0, "skipped_status": 0, "skipped_bom": 0}
        failed = {s.strip().casefold() for s in statuses}
        for node in self.root.findall("./{*}Board/{*}Component"):
            self.selection["total"] += 1
            if node.get("Status", "").strip().casefold() not in failed:
                self.selection["skipped_status"] += 1
                continue
            if node.get("Type", "").strip() not in allowed_types:
                self.selection["skipped_bom"] += 1
                continue
            regions: list[ImageRegion] = []
            for image in node.findall("./{*}CompImage/{*}Image"):
                try:
                    crop = CropBox.from_attributes(image.attrib)
                except ValueError as exc:
                    raise ValueError(f"Component {node.get('CompName')}: {exc}") from exc
                # Online AOI emits one PicPath per Image; older exports pair paths.
                attributes = ("PicPath1", "PicPath2") if (
                    "PicPath1" in image.attrib or "PicPath2" in image.attrib
                ) else ("PicPath",)
                for attr in attributes:
                    name = image_basename(image.get(attr, ""))
                    if Path(name).suffix.lower() not in extensions:
                        raise ValueError(f"Unsupported image extension: {name}")
                    regions.append(ImageRegion(name, crop))
            if not regions:
                raise ValueError(f"Component {node.get('CompName')} has no CompImage/Image")
            if len(regions) < 2:
                raise ValueError(f"Component {node.get('CompName')} requires at least two light-source images")
            self.components.append(Component(node, regions))
            self.selection["selected"] += 1

    @property
    def regions(self) -> list[ImageRegion]:
        return list(dict.fromkeys(region for comp in self.components for region in comp.images))

    @property
    def image_names(self) -> list[str]:
        # Files are tracked once for readiness/archive; different ROIs remain separate inference jobs.
        return list(dict.fromkeys(region.filename for region in self.regions))

    def result(self, verdicts: dict[str, bool]) -> bytes:
        for component in self.components:
            # All unique (file, crop) pairs are inferred before component aggregation.
            ok = all(verdicts[region.key] for region in component.images)
            component.node.set("AI_result", "OK" if ok else "NG")
        return ET.tostring(self.root, encoding="utf-8", xml_declaration=True)


def referenced_names(data: bytes) -> set[str]:
    root = ET.fromstring(data)
    return {
        image_basename(image.get(attr, ""))
        for image in root.findall("./{*}Board/{*}Component/{*}CompImage/{*}Image")
        for attr in ("PicPath", "PicPath1", "PicPath2") if image.get(attr)
    }
