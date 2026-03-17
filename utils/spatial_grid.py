"""
utils/spatial_grid.py
Aspect-aware 6x6 spatial grid for Manim scene layout analysis and VLM Critic.

Supports both landscape (16:9) and portrait (9:16) aspect ratios.
All geometry is computed dynamically from the aspect — no hardcoded constants.

Landscape 16:9:
  Manim canvas: 14 wide x 8 tall (x: -7->+7, y: -4->+4)
  Natural layout: TITLE + MAIN + SIDEBAR (horizontal flow)

Portrait 9:16:
  Manim canvas: 8 wide x 14 tall (x: -4->+4, y: -7->+7)
  Natural layout: TITLE + UPPER_MAIN + LOWER_MAIN + FOOTER (vertical stack)
  SIDEBAR is excluded — too narrow in portrait mode.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Optional

GRID_COLS = 6
GRID_ROWS = 6

_LANDSCAPE = {"frame_w": 14.0, "frame_h": 8.0,  "px_w": 1920, "px_h": 1080}
_PORTRAIT  = {"frame_w":  8.0, "frame_h": 14.0, "px_w": 1080, "px_h": 1920}


def _geometry(aspect: str) -> dict:
    return _PORTRAIT if aspect == "9:16" else _LANDSCAPE


@dataclass(frozen=True)
class Zone:
    name:          str
    description:   str
    row_start:     int
    row_end:       int
    col_start:     int
    col_end:       int
    manim_center:  tuple
    manim_topleft: tuple
    pixel_bbox:    tuple


def _build_zones(aspect: str) -> dict:
    g      = _geometry(aspect)
    fw, fh = g["frame_w"], g["frame_h"]
    pw, ph = g["px_w"],    g["px_h"]
    cw     = fw / GRID_COLS
    ch     = fh / GRID_ROWS
    pcw    = pw / GRID_COLS
    pch    = ph / GRID_ROWS

    def mc(r0, r1, c0, c1):
        x = -(fw / 2) + ((c0 + c1) / 2 + 0.5) * cw
        y =  (fh / 2) - ((r0 + r1) / 2 + 0.5) * ch
        return round(x, 2), round(y, 2)

    def tl(r0, c0):
        return round(-(fw / 2) + c0 * cw, 2), round((fh / 2) - r0 * ch, 2)

    def bb(r0, r1, c0, c1):
        return int(c0*pcw), int(r0*pch), int((c1+1)*pcw), int((r1+1)*pch)

    def z(name, desc, r0, r1, c0, c1):
        return Zone(name=name, description=desc,
                    row_start=r0, row_end=r1, col_start=c0, col_end=c1,
                    manim_center=mc(r0,r1,c0,c1),
                    manim_topleft=tl(r0,c0),
                    pixel_bbox=bb(r0,r1,c0,c1))

    if aspect == "9:16":
        zones = [
            z("TITLE",       "Top banner — scene title or chapter heading",                  0,0,0,5),
            z("SUBTITLE",    "Second row — subtitle, tagline, or equation header",           1,1,0,5),
            z("FOOTER",      "Bottom banner — exam tip, recap line, or citation",            5,5,0,5),
            z("MAIN",        "Primary content — full-width central area (rows 2-3)",         2,3,0,5),
            z("UPPER_MAIN",  "Upper content area — first element in vertical stack",         1,2,0,5),
            z("LOWER_MAIN",  "Lower content area — second element or answer reveal",         3,4,0,5),
            z("UPPER_HALF",  "Top half — first panel in 2-panel vertical split",            0,2,0,5),
            z("LOWER_HALF",  "Bottom half — second panel in 2-panel vertical split",        3,5,0,5),
            z("UPPER_LEFT",  "Upper-left quadrant — first panel in 2x2 layout",            1,2,0,2),
            z("UPPER_RIGHT", "Upper-right quadrant — second panel in 2x2 layout",          1,2,3,5),
            z("LOWER_LEFT",  "Lower-left quadrant — third panel in 2x2 layout",            3,4,0,2),
            z("LOWER_RIGHT", "Lower-right quadrant — fourth panel in 2x2 layout",          3,4,3,5),
            z("CENTER",      "Absolute canvas centre — isolated focus element",              2,3,1,4),
        ]
    else:
        zones = [
            z("TITLE",        "Top banner — scene title or chapter heading",                 0,0,0,5),
            z("SUBTITLE",     "Second row — subtitle, tagline, or equation header",          1,1,0,3),
            z("FOOTER",       "Bottom banner — exam tip, recap line, or citation",           5,5,0,5),
            z("MAIN",         "Primary content area — central diagram, proof, or animation", 1,4,0,3),
            z("UPPER_MAIN",   "Upper half of main — first half of two-part layout",          1,2,0,3),
            z("LOWER_MAIN",   "Lower half of main — second half, answer, or detail",         3,4,0,3),
            z("SIDEBAR",      "Right sidebar — callout, legend, annotation",                 1,4,4,5),
            z("UPPER_SIDEBAR","Upper sidebar — primary callout or highlighted fact",          1,2,4,5),
            z("LOWER_SIDEBAR","Lower sidebar — secondary callout or follow-up note",          3,4,4,5),
            z("UPPER_LEFT",   "Upper-left quadrant — first panel in 2x2 layout",            1,2,0,2),
            z("UPPER_RIGHT",  "Upper-right quadrant — second panel in 2x2 layout",          1,2,3,5),
            z("LOWER_LEFT",   "Lower-left quadrant — third panel in 2x2 layout",            3,4,0,2),
            z("LOWER_RIGHT",  "Lower-right quadrant — fourth panel in 2x2 layout",          3,4,3,5),
            z("CENTER",       "Absolute canvas centre — isolated focus element",              2,3,2,3),
        ]

    return {z.name: z for z in zones}


# Legacy-compatible module-level defaults (landscape)
ZONES:      dict = _build_zones("16:9")
ZONE_NAMES: list = list(ZONES.keys())


def get_zones(aspect: str = "16:9") -> dict:
    return _build_zones(aspect)


def zone_to_manim_position(zone_name: str, aspect: str = "16:9") -> str:
    zone = get_zones(aspect).get(zone_name)
    if not zone:
        return ".move_to(ORIGIN)"
    x, y = zone.manim_center
    return f".move_to(np.array([{x}, {y}, 0]))"


def zone_to_shift_hint(from_zone: str, to_zone: str, aspect: str = "16:9") -> str:
    zones = get_zones(aspect)
    fz, tz = zones.get(from_zone), zones.get(to_zone)
    if not fz or not tz:
        return f"Move element to zone {to_zone}"
    return (f"Move the element from {from_zone} center {fz.manim_center} "
            f"to {to_zone} center {tz.manim_center} ({tz.description})")


def zone_at_pixel(px: int, py: int, aspect: str = "16:9") -> Zone:
    g     = _geometry(aspect)
    zones = get_zones(aspect)
    col   = min(int(px / (g["px_w"] / GRID_COLS)), GRID_COLS - 1)
    row   = min(int(py / (g["px_h"] / GRID_ROWS)), GRID_ROWS - 1)
    for zone in zones.values():
        if zone.row_start <= row <= zone.row_end and zone.col_start <= col <= zone.col_end:
            return zone
    return zones.get("MAIN") or list(zones.values())[0]


def zone_manifest(aspect: str = "16:9") -> str:
    zones = get_zones(aspect)
    label = "PORTRAIT 9:16" if aspect == "9:16" else "LANDSCAPE 16:9"
    lines = [f"ZONE LAYOUT ({label})", "ZONE          | MANIM CENTER   | DESCRIPTION", "-"*72]
    for zone in zones.values():
        x, y = zone.manim_center
        lines.append(f"{zone.name:<14}| ({x:+.2f}, {y:+.2f})    | {zone.description}")
    return "\n".join(lines)


def frame_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def draw_grid_overlay(image_bytes: bytes, aspect: str = "16:9", alpha: int = 140) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    g   = _geometry(aspect)
    pw, ph = g["px_w"], g["px_h"]
    zones  = get_zones(aspect)
    pcw    = pw / GRID_COLS
    pch    = ph / GRID_ROWS

    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    if img.size != (pw, ph):
        img = img.resize((pw, ph), Image.LANCZOS)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    lc = (255, 255, 0, alpha)
    for c in range(1, GRID_COLS):
        x = int(c * pcw)
        draw.line([(x, 0), (x, ph)], fill=lc, width=2)
    for r in range(1, GRID_ROWS):
        y = int(r * pch)
        draw.line([(0, y), (pw, y)], fill=lc, width=2)

    try:
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        fz = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        fs = fz = ImageFont.load_default()

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            draw.text((int(c*pcw)+6, int(r*pch)+4), f"({r},{c})",
                      fill=(220,220,220,200), font=fs)

    label_zones = (["TITLE","MAIN","LOWER_MAIN","FOOTER","CENTER"]
                   if aspect == "9:16"
                   else ["TITLE","MAIN","SIDEBAR","FOOTER","CENTER"])
    for zname in label_zones:
        zone = zones.get(zname)
        if not zone:
            continue
        bx0,by0,bx1,by1 = zone.pixel_bbox
        cx, cy = (bx0+bx1)//2, (by0+by1)//2
        tw = len(zname)*14+16
        draw.rectangle([(cx-tw//2,cy-15),(cx+tw//2,cy+15)], fill=(30,30,30,180))
        draw.text((cx-tw//2+8,cy-11), zname, fill=(80,220,255,230), font=fz)

    composited = Image.alpha_composite(img, overlay).convert("RGB")
    buf = io.BytesIO()
    composited.save(buf, format="PNG", optimize=False)
    return buf.getvalue()