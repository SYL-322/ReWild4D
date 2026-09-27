"""Split the PPT-exported section videos at their slide boundaries."""

from __future__ import annotations

import argparse
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile


SITE_DIR = Path(__file__).resolve().parents[1]
VIDEO_DIR = SITE_DIR / "static" / "videos"
SLIDE_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"

SECTIONS = (
    (
        "results-gallery",
        "ECCV_Results_Applications_Limitations.pptx",
        1,
        ("impala", "sheep", "horse", "cow", "gold-fish", "other-animals", "giraffe-and-zebra"),
    ),
    (
        "comparison",
        "ECCV_Comparisons.pptx",
        1,
        ("davis", "online", "aptv2-1", "aptv2-2", "artemis-1", "artemis-2"),
    ),
    (
        "application",
        "ECCV_Results_Applications_Limitations.pptx",
        9,
        ("part-level-editing", "temporal-and-appearance-editing"),
    ),
    (
        "limitations",
        "ECCV_Results_Applications_Limitations.pptx",
        12,
        ("moving-camera", "limited-view-coverage", "occlusion"),
    ),
)


def slide_lengths(deck: Path, first: int, count: int) -> list[float]:
    with ZipFile(deck) as archive:
        lengths = []
        for number in range(1, first + count):
            slide = ET.fromstring(archive.read(f"ppt/slides/slide{number}.xml"))
            transition = slide.find(f".//{{{SLIDE_NS}}}transition")
            if transition is None or "advTm" not in transition.attrib:
                raise ValueError(f"Missing slide advance time: {deck.name}, slide {number}")
            lengths.append(int(transition.attrib["advTm"]) / 1000)
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", required=True, type=Path)
    parser.add_argument("--pptx-dir", type=Path, default=SITE_DIR.parent)
    args = parser.parse_args()

    for section, deck_name, first_slide, names in SECTIONS:
        lengths = slide_lengths(args.pptx_dir / deck_name, first_slide, len(names) + 1)
        source = VIDEO_DIR / f"{section}.mp4"
        destination = VIDEO_DIR / section
        destination.mkdir(exist_ok=True)
        start = lengths[first_slide - 1] + 0.15

        for name, duration in zip(names, lengths[first_slide:]):
            output = destination / f"{name}.mp4"
            print(f"{section}/{name}: {start:.3f}s, {duration - 0.3:.3f}s", flush=True)
            subprocess.run(
                [
                    str(args.ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", f"{start:.3f}", "-i", str(source),
                    "-t", f"{duration - 0.3:.3f}", "-an", "-c:v", "libx264",
                    "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(output),
                ],
                check=True,
            )
            start += duration


if __name__ == "__main__":
    main()
