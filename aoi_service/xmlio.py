"""Read AOI XML and update only Component.AI_result."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .files import image_basename
from .regions import CropBox, ImageRegion


@dataclass
class Component:
    node: ET.Element
    images: list[ImageRegion]


class Document:
    def __init__(self, data: bytes, statuses: list[str], extensions: list[str]):
        if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
            raise ValueError("DTD/entity declarations are not supported")
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True, insert_pis=True))
        self.root = ET.fromstring(data, parser=parser)
        if self.root.tag.rsplit("}", 1)[-1] != "Panel":
            raise ValueError("Expected an AOI Panel XML document")
        self.components: list[Component] = []
        failed = {s.upper() for s in statuses}
        for node in self.root.findall("./{*}Board/{*}Component"):
            if node.get("Status", "").upper() not in failed:
                continue
            regions: list[ImageRegion] = []
            for image in node.findall("./{*}CompImage/{*}Image"):
                try:
                    crop = CropBox.from_attributes(image.attrib)
                except ValueError as exc:
                    raise ValueError(f"Component {node.get('CompName')}: {exc}") from exc
                for attr in ("PicPath1", "PicPath2"):
                    name = image_basename(image.get(attr, ""))
                    if Path(name).suffix.lower() not in extensions:
                        raise ValueError(f"Unsupported image extension: {name}")
                    regions.append(ImageRegion(name, crop))
            if not regions:
                raise ValueError(f"Component {node.get('CompName')} has no CompImage/Image")
            self.components.append(Component(node, regions))

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
        for attr in ("PicPath1", "PicPath2") if image.get(attr)
    }
