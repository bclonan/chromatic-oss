"""Render the documentation share card. Requires Pillow and a local sans font."""
from pathlib import Path
import colorsys
import math

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def font(size, bold=False):
    candidates = ["arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    raise RuntimeError("Install Arial or DejaVu Sans to reproduce the share card")


def main():
    image = Image.new("RGB", (1200, 630), "#f8f7f2")
    draw = ImageDraw.Draw(image)
    ink, muted = "#182b29", "#53635c"
    draw.text((60, 44), "CHROMATIC", font=font(22, True), fill=ink)
    draw.text((60, 103), "AN EXPERIMENT BY BRAD CLONAN", font=font(15), fill=muted)
    draw.text((56, 164), "What if weights", font=font(65, True), fill=ink)
    draw.text((56, 239), "looked like color?", font=font(65, True), fill=ink)
    draw.text((60, 346), "Color tokens. Exact hue arithmetic.", font=font(25), fill=muted)
    draw.text((60, 384), "Neural network ideas, tested in the open.", font=font(25), fill=muted)
    # A finite wheel beside a numeric-to-color network sketch.
    center = (959, 221)
    for index in range(120):
        angle = index / 120 * 2 * math.pi
        color = tuple(round(c * 255) for c in colorsys.hsv_to_rgb(index / 120, .63, .8))
        x, y = center[0] + math.cos(angle) * 125, center[1] + math.sin(angle) * 125
        draw.ellipse((x-7, y-7, x+7, y+7), fill=color)
    draw.text((903, 201), "RGBD", font=font(29, True), fill=ink)
    draw.text((915, 242), "4 bytes", font=font(20), fill=muted)
    left = [(813, 402), (813, 474)]
    middle = [(968, 382), (968, 442), (968, 502)]
    for a in left:
        for b in middle:
            draw.line([a, b], fill="#a0b5a9", width=2)
    for b in middle:
        draw.line([b, (1110, 442)], fill="#a0b5a9", width=2)
    for point, text in zip(left, ["0.8", "-0.4"]):
        x, y = point
        draw.ellipse((x-26,y-26,x+26,y+26),fill="#f8f7f2",outline=ink,width=2)
        draw.text((x-20,y-10),text,font=font(17),fill=ink)
    for point,color in zip(middle,["#c77355","#7d96c4","#81ad75"]):
        x,y=point
        draw.ellipse((x-19,y-19,x+19,y+19),fill=color,outline=ink,width=2)
    draw.ellipse((1090,422,1130,462),fill="#c09fc8",outline=ink,width=2)
    draw.line((60,548,1140,548),fill="#ccd7cb",width=1)
    draw.text((60,574),"chromatic-hue.netlify.app",font=font(21),fill=ink)
    draw.text((788,579),"EXPERIMENTAL / VISUAL EXPLAINER",font=font(13),fill=muted)
    destination = ROOT / "docs/site/chromatic-social.png"
    image.save(destination, optimize=True)
    print(destination.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
