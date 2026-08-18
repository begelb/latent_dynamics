"""Recolor a vector Chafee--Infante PDF without changing its geometry.

This is used for persisted paper figures whose exact surviving render artifacts
should not be geometrically reconstructed. The script changes only known RGB
drawing operators or indexed-image palette entries and leaves paths, text,
page boxes, pixel indices, and every other PDF object untouched.

The ``reference_d2`` mapping translates the default matplotlib palette of the
coauthor-provided d=2 adaptive computation into the paper-wide palette; the
published fine d=2 Morse graph and Morse sets panels
(``figures/chafee_infante/ci_morse_graph.pdf`` and ``ci_morse_sets.pdf``) are
this recoloring applied to the author-provided PDFs, for which no DOT/CSV
sources survive. The ``d1-coarse`` mapping greys the connecting set of a
coarse rendering.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import ByteStringObject


OLD_PALETTE = (
    "#1F77B4",
    "#E6550D",
    "#31A354",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
)
NEW_PALETTE = (
    "#DC267F",  # raw node 0 is the physical positive attractor
    "#FFB000",  # raw node 1 is the physical negative attractor
    "#648FFF",
    "#FE6100",
    "#785EF0",
    "#008080",
    "#FCC2E8",
)


def _rgb(hex_color: str) -> tuple[float, float, float]:
    return tuple(
        int(hex_color[index : index + 2], 16) / 255.0
        for index in (1, 3, 5)
    )


COLOR_MAPS = {
    "reference_d2": dict(
        zip(map(_rgb, OLD_PALETTE), map(_rgb, NEW_PALETTE), strict=True)
    ),
    "d1-coarse": {_rgb("#648FFF"): _rgb("#7F7F7F")},
}
COLOR_OPERATOR = re.compile(
    rb"(?P<r>[+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    rb"(?P<s1>\s+)"
    rb"(?P<g>[+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    rb"(?P<s2>\s+)"
    rb"(?P<b>[+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    rb"(?P<s3>\s+)"
    rb"(?P<operator>rg|RG)\b"
)


def _match_color(
    values: tuple[float, float, float],
    color_map: dict[tuple[float, float, float], tuple[float, float, float]],
) -> tuple[float, float, float] | None:
    for original, replacement in color_map.items():
        if max(abs(left - right) for left, right in zip(values, original, strict=True)) < 1e-6:
            return replacement
    return None


def _recolor_indexed_images(
    page,
    color_map: dict[tuple[float, float, float], tuple[float, float, float]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    resources = page.get("/Resources")
    if resources is None:
        return counts
    xobjects = resources.get("/XObject")
    if xobjects is None:
        return counts
    byte_map = {
        tuple(round(channel * 255) for channel in original): tuple(
            round(channel * 255) for channel in replacement
        )
        for original, replacement in color_map.items()
    }
    for reference in xobjects.values():
        image = reference.get_object()
        color_space = image.get("/ColorSpace")
        if (
            image.get("/Subtype") != "/Image"
            or not isinstance(color_space, list)
            or len(color_space) != 4
            or color_space[0] != "/Indexed"
            or not isinstance(color_space[3], bytes)
        ):
            continue
        lookup = bytearray(color_space[3])
        for offset in range(0, len(lookup), 3):
            original = tuple(lookup[offset : offset + 3])
            replacement = byte_map.get(original)
            if replacement is None:
                continue
            lookup[offset : offset + 3] = bytes(replacement)
            counts[f"indexed:{original}"] += 1
        color_space[3] = ByteStringObject(bytes(lookup))
    return counts


def _pdf_number(value: float) -> bytes:
    return f"{value:.10f}".rstrip("0").rstrip(".").encode("ascii")


def _recolor_content_streams(
    page,
    color_map: dict[tuple[float, float, float], tuple[float, float, float]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    contents = page.get("/Contents")
    if contents is None:
        return counts
    references = contents if isinstance(contents, list) else (contents,)
    for reference in references:
        stream = reference.get_object()

        def replace(match: re.Match[bytes]) -> bytes:
            values = tuple(
                float(match.group(name))
                for name in ("r", "g", "b")
            )
            replacement = _match_color(values, color_map)
            if replacement is None:
                return match.group(0)
            operator = match.group("operator").decode("ascii")
            counts[f"{operator}:{values}"] += 1
            return b"".join(
                (
                    _pdf_number(replacement[0]),
                    match.group("s1"),
                    _pdf_number(replacement[1]),
                    match.group("s2"),
                    _pdf_number(replacement[2]),
                    match.group("s3"),
                    match.group("operator"),
                )
            )

        original = stream.get_data()
        updated = COLOR_OPERATOR.sub(replace, original)
        if updated != original:
            stream.set_data(updated)
    return counts


def recolor_pdf(
    source: Path,
    destination: Path,
    color_map: dict[tuple[float, float, float], tuple[float, float, float]],
) -> Counter[str]:
    """Write a geometry-identical PDF whose known palette operators are replaced."""

    writer = PdfWriter(clone_from=source)
    counts: Counter[str] = Counter()
    for page in writer.pages:
        counts.update(_recolor_content_streams(page, color_map))
        counts.update(_recolor_indexed_images(page, color_map))

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        writer.write(output)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--mapping",
        choices=tuple(COLOR_MAPS),
        default="reference_d2",
        help="known source-palette mapping to apply",
    )
    args = parser.parse_args()
    counts = recolor_pdf(
        args.source.resolve(),
        args.destination.resolve(),
        COLOR_MAPS[args.mapping],
    )
    if sum(counts.values()) == 0:
        raise ValueError(f"no known Chafee palette operators found in {args.source}")
    print(
        json.dumps(
            {
                "source": str(args.source.resolve()),
                "destination": str(args.destination.resolve()),
                "mapping": args.mapping,
                "operator_replacements": dict(counts),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
