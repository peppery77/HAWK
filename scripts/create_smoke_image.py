#!/usr/bin/env python3
"""Create a deterministic, redistribution-safe image for smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("assets/smoke_test.png"))
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (768, 512), "#8fd3ff")
    draw = ImageDraw.Draw(image)

    # Sun and ground.
    draw.ellipse((610, 40, 700, 130), fill="#ffd54f", outline="#f6a800", width=4)
    draw.rectangle((0, 310, 768, 512), fill="#71b85b")

    # Two trees.
    for x, scale in ((105, 1.0), (620, 0.9)):
        trunk_width = int(42 * scale)
        draw.rectangle(
            (x - trunk_width // 2, 205, x + trunk_width // 2, 390),
            fill="#805333",
        )
        crown = int(92 * scale)
        draw.ellipse(
            (x - crown, 92, x + crown, 92 + 2 * crown),
            fill="#2f8f4e",
            outline="#226d3b",
            width=5,
        )

    # A red picnic table centered between the trees.
    draw.polygon([(285, 290), (500, 290), (530, 325), (255, 325)], fill="#b74433")
    draw.rectangle((280, 338, 505, 365), fill="#cf5945")
    draw.line((315, 320, 275, 460), fill="#6f382d", width=18)
    draw.line((470, 320, 515, 460), fill="#6f382d", width=18)
    draw.line((335, 360, 315, 460), fill="#6f382d", width=14)
    draw.line((450, 360, 470, 460), fill="#6f382d", width=14)

    # Small blue cup on the tabletop.
    draw.rectangle((380, 258, 415, 293), fill="#2e75b6", outline="#194c7a", width=3)
    draw.ellipse((380, 250, 415, 267), fill="#61a5d8", outline="#194c7a", width=3)

    image.save(output)
    print(output)


if __name__ == "__main__":
    main()
