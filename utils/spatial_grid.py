"""
utils/spatial_grid.py
Aspect-aware 6x6 spatial grid for Manim scene layout analysis and VLM Critic.

Supports both landscape (16:9) and portrait (9:16) aspect ratios.
All geometry is computed dynamically from the aspect — no hardcoded constants.

Landscape 16:9:
  Manim canvas: 14 wide x 8 tall (x: -7->+7, y: -4->+4)
  Natural layout: TITLE + MAIN + SIDEBAR (horizontal flow)

Portrait 9:16 (TikTok/Reels/Shorts):
  Manim canvas: 8 wide x 14 tall (x: -4->+4, y: -7->+7)
  TikTok-safe canvas: cols 0-3 (x: -4->+1.33), rows 0-4 (y: -4.67->+7)

  FORBIDDEN ZONES (TikTok UI overlays — never place content here):
    TIKTOK_BUTTONS: right edge (cols 4-5) — like/comment/share/follow buttons
    TIKTOK_TITLE:   bottom row (row 5)    — username, song, caption bar

  Safe layout: TITLE (top) + UPPER_MAIN + LOWER_MAIN (vertical stack, left of buttons)
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
        # ── TikTok-aware portrait zones ──────────────────────────────────────
        # TikTok UI overlays occupy two regions that must NEVER contain content:
        #
        #   TIKTOK_BUTTONS (cols 4-5, rows 1-4):
        #     Like, comment, share, follow buttons on the right edge.
        #     Approx x > +2.67 in Manim units. Keep ALL content in cols 0-3.
        #
        #   TIKTOK_TITLE (row 5, cols 0-3):
        #     Username, song title, and caption text at the bottom-left.
        #     Approx y < -4.67 in Manim units. Keep ALL content in rows 0-4.
        #
        # Safe canvas: cols 0-3 (x ∈ [-4, +1.33]), rows 0-4 (y ∈ [-4.67, +7])
        # All content zones below are capped to this safe region.
        zones = [
            # ── Content zones (all within TikTok-safe bounds) ─────────────────
            # Row 0 is reserved — TikTok's search bar overlays the top ~130px
            # (≈ top 12% of screen = row 0 in a 6-row grid). TITLE starts at
            # row 1 so it sits below the search bar with natural breathing room.
            z("TITLE",       "Title zone — below TikTok search bar (row 1)",                 1,1,0,3),
            z("SUBTITLE",    "Subtitle / equation header (row 2)",                            2,2,0,3),
            z("MAIN",        "Primary content area — central animation zone",                 1,3,0,3),
            z("UPPER_MAIN",  "Upper animation area — first beat in vertical stack",           1,2,0,3),
            z("LOWER_MAIN",  "Lower animation area — second beat, above TikTok title bar",   3,4,0,3),
            z("CENTER",      "Canvas centre within safe zone",                                2,3,1,3),
            z("UPPER_LEFT",  "Upper-left quadrant",                                           1,2,0,1),
            z("UPPER_RIGHT", "Upper-right quadrant (safe — stays left of TikTok buttons)",   1,2,2,3),
            z("LOWER_LEFT",  "Lower-left quadrant",                                           3,4,0,1),
            z("LOWER_RIGHT", "Lower-right quadrant (safe — stays left of TikTok buttons)",   3,4,2,3),
            # ── Forbidden zones — marker only, never place content here ───────
            z("TIKTOK_SEARCH", "FORBIDDEN — TikTok search bar (top row, ~130px)",            0,0,0,5),
            z("TIKTOK_BUTTONS","FORBIDDEN — TikTok like/share/follow buttons (right edge)",  1,4,4,5),
            z("TIKTOK_TITLE",  "FORBIDDEN — TikTok username/caption bar (bottom)",           5,5,0,5),
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


def _grid_cell_to_manim(ref: str, aspect: str = "16:9") -> tuple[float, float] | None:
    """
    Parse a raw grid cell reference like '(1,5)' or '1,5' into a Manim (x,y)
    coordinate. Returns None if the string doesn't match the pattern.
    Gemini often returns grid coordinates instead of zone names because the
    frame overlay only labels 5 landmark zones; this handles that gracefully.
    """
    import re
    m = re.match(r"[\(\[]?\s*(\d+)\s*[,;]\s*(\d+)\s*[\)\]]?", ref.strip())
    if not m:
        return None
    row, col = int(m.group(1)), int(m.group(2))

    g       = _geometry(aspect)
    fw, fh  = g["frame_w"], g["frame_h"]
    cw, ch  = fw / GRID_COLS, fh / GRID_ROWS

    # Clamp to valid grid range (0..GRID_ROWS-1, 0..GRID_COLS-1)
    row = max(0, min(row, GRID_ROWS - 1))
    col = max(0, min(col, GRID_COLS - 1))

    # Cell centre in Manim coordinates
    x = round(-(fw / 2) + (col + 0.5) * cw, 2)
    y = round( (fh / 2) - (row + 0.5) * ch, 2)

    # Clamp to safe zone (avoid edges where Manim clips content).
    # Portrait gets a larger left margin (0.5 units) because the physical
    # screen edge clips text — as seen in production. Landscape uses the
    # standard half-cell margin on both sides.
    if aspect == "9:16":
        left_margin  = 0.5          # physical screen edge safety
        right_margin = cw * 0.6    # TikTok buttons buffer
    else:
        left_margin  = cw * 0.6
        right_margin = cw * 0.6
    safe_y = fh / 2 - ch * 0.6
    x = max(-(fw / 2) + left_margin, min(x, (fw / 2) - right_margin))
    y = max(-safe_y, min(y, safe_y))
    return x, y


def zone_to_manim_position(zone_name: str, aspect: str = "16:9") -> str:
    """
    Resolve a zone name OR a raw grid coordinate like '(1,5)' to a Manim
    .move_to() call. Falls back to grid-cell math if the zone name is not
    in the named-zone registry — which happens when Gemini returns raw cell
    references from the frame overlay instead of zone names.
    """
    # Try named zone first
    zone = get_zones(aspect).get(zone_name)
    if zone:
        x, y = zone.manim_center
        return f".move_to(np.array([{x}, {y}, 0]))"

    # Try to parse as a raw grid cell reference e.g. "(1,5)"
    coords = _grid_cell_to_manim(zone_name, aspect)
    if coords:
        x, y = coords
        return f".move_to(np.array([{x}, {y}, 0]))"

    # True fallback — zone name unrecognised and not a grid ref
    # Use MAIN zone centre rather than ORIGIN to avoid centre pile-up
    main = get_zones(aspect).get("MAIN")
    if main:
        x, y = main.manim_center
        return f".move_to(np.array([{x}, {y}, 0]))"
    return ".move_to(ORIGIN)"


def zone_to_shift_hint(from_zone: str, to_zone: str, aspect: str = "16:9") -> str:
    zones = get_zones(aspect)
    fz, tz = zones.get(from_zone), zones.get(to_zone)

    # Resolve raw grid refs if named zones not found
    from_coords = fz.manim_center if fz else _grid_cell_to_manim(from_zone, aspect)
    to_coords   = tz.manim_center if tz else _grid_cell_to_manim(to_zone, aspect)
    to_desc     = tz.description  if tz else f"grid cell {to_zone}"

    if not from_coords or not to_coords:
        return f"Move element to zone {to_zone}"
    return (f"Move the element from {from_zone} center {from_coords} "
            f"to {to_zone} center {to_coords} ({to_desc})")


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

    label_zones = (["TITLE","MAIN","LOWER_MAIN","TIKTOK_BUTTONS","TIKTOK_TITLE"]
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
