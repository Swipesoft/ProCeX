"""
utils/spatial_grid.py
Domain-agnostic 6×6 spatial grid for Manim scene layout analysis and
VLM Critic feedback.

Manim default coordinate system:
  X: -7.0 (left) → +7.0 (right)   frame_width  = 14
  Y: -4.0 (bottom) → +4.0 (top)   frame_height = 8

Grid layout (6 rows × 6 cols):
  Cell width  = 14 / 6 ≈ 2.33 Manim units
  Cell height =  8 / 6 ≈ 1.33 Manim units

  Col:  0      1      2      3      4      5
  Row 0 [──────────── TITLE ────────────────]
  Row 1 [────── UPPER ──────][── SIDEBAR ───]
  Row 2 [────── MAIN  ──────][── SIDEBAR ───]
  Row 3 [────── MAIN  ──────][── SIDEBAR ───]
  Row 4 [────── LOWER ──────][── SIDEBAR ───]
  Row 5 [─────────── FOOTER ─────────────── ]

Named zones are intentionally generic — they describe spatial intent, not
subject matter, so they work equally well for math, CS, ML, nursing, or
social-media content.

Every zone exposes:
  - manim_center (x, y)       → where to place element in Manim coords
  - manim_top_left (x, y)     → useful for .to_corner() / .shift() recipes
  - pixel_bbox (x0, y0, x1, y1) → bounding box in a 1920×1080 frame image
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import base64
import io

# ── Manim canvas constants ─────────────────────────────────────────────────────
FRAME_W     = 14.0    # Manim units, x: -7 → +7
FRAME_H     =  8.0    # Manim units, y: -4 → +4
GRID_COLS   =  6
GRID_ROWS   =  6
CELL_W      = FRAME_W / GRID_COLS   # ≈ 2.333
CELL_H      = FRAME_H / GRID_ROWS   # ≈ 1.333

# ── Pixel canvas (for frame overlay rendering) ─────────────────────────────────
PX_W        = 1920
PX_H        = 1080
PX_CELL_W   = PX_W / GRID_COLS     # 320 px
PX_CELL_H   = PX_H / GRID_ROWS     # 180 px


# ── Helper: cell → Manim coordinate conversions ───────────────────────────────

def _cell_center_manim(row: float, col: float) -> tuple[float, float]:
    """Return the Manim (x, y) of the center of a (possibly fractional) cell."""
    x = -7.0 + (col + 0.5) * CELL_W
    y =  4.0 - (row + 0.5) * CELL_H
    return round(x, 2), round(y, 2)


def _cell_topleft_manim(row: int, col: int) -> tuple[float, float]:
    x = -7.0 + col * CELL_W
    y =  4.0 - row * CELL_H
    return round(x, 2), round(y, 2)


def _span_center_manim(
    row_start: int, row_end: int, col_start: int, col_end: int
) -> tuple[float, float]:
    """Center of a rectangular span (inclusive end rows/cols)."""
    mid_row = (row_start + row_end) / 2
    mid_col = (col_start + col_end) / 2
    return _cell_center_manim(mid_row, mid_col)


def _cell_topleft_px(row: int, col: int) -> tuple[int, int]:
    return int(col * PX_CELL_W), int(row * PX_CELL_H)


def _span_bbox_px(
    row_start: int, row_end: int, col_start: int, col_end: int
) -> tuple[int, int, int, int]:
    x0 = int(col_start * PX_CELL_W)
    y0 = int(row_start * PX_CELL_H)
    x1 = int((col_end + 1) * PX_CELL_W)
    y1 = int((row_end + 1) * PX_CELL_H)
    return x0, y0, x1, y1


# ── Zone definition ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Zone:
    name:           str
    description:    str          # plain-English for VLM prompt
    row_start:      int
    row_end:        int          # inclusive
    col_start:      int
    col_end:        int          # inclusive
    manim_center:   tuple        # (x, y) Manim units
    manim_topleft:  tuple        # (x, y) Manim units — top-left corner
    pixel_bbox:     tuple        # (x0, y0, x1, y1) in 1920×1080 px


def _make_zone(name, description, r0, r1, c0, c1) -> Zone:
    return Zone(
        name          = name,
        description   = description,
        row_start     = r0,
        row_end       = r1,
        col_start     = c0,
        col_end       = c1,
        manim_center  = _span_center_manim(r0, r1, c0, c1),
        manim_topleft = _cell_topleft_manim(r0, c0),
        pixel_bbox    = _span_bbox_px(r0, r1, c0, c1),
    )


# ── Zone registry ──────────────────────────────────────────────────────────────
# 14 named zones covering all meaningful spatial positions.
# Intentionally subject-agnostic: "SIDEBAR" works for a CS dependency graph,
# a nursing BowTie arm, a maths side annotation, or a social-media stat pull-out.

ZONES: dict[str, Zone] = {z.name: z for z in [
    # Full-width banner rows
    _make_zone("TITLE",       "Top banner — scene title or chapter heading",
               0, 0,  0, 5),
    _make_zone("SUBTITLE",    "Second row — subtitle, tagline, or equation header",
               1, 1,  0, 3),
    _make_zone("FOOTER",      "Bottom banner — exam tip, recap line, or source citation",
               5, 5,  0, 5),

    # Main content area (left 4 cols, rows 1–4)
    _make_zone("MAIN",        "Primary content area — central diagram, proof, or animation",
               1, 4,  0, 3),
    _make_zone("UPPER_MAIN",  "Upper half of main area — first half of a two-part layout",
               1, 2,  0, 3),
    _make_zone("LOWER_MAIN",  "Lower half of main area — second half, answer reveal, or detail",
               3, 4,  0, 3),

    # Sidebar (right 2 cols, rows 1–4)
    _make_zone("SIDEBAR",     "Right sidebar — callout box, legend, annotation, or secondary info",
               1, 4,  4, 5),
    _make_zone("UPPER_SIDEBAR","Upper sidebar — primary callout or highlighted fact",
               1, 2,  4, 5),
    _make_zone("LOWER_SIDEBAR","Lower sidebar — secondary callout or follow-up note",
               3, 4,  4, 5),

    # Quadrants (for two-panel layouts)
    _make_zone("UPPER_LEFT",  "Upper-left quadrant — first panel in a 2×2 layout",
               1, 2,  0, 2),
    _make_zone("UPPER_RIGHT", "Upper-right quadrant — second panel in a 2×2 layout",
               1, 2,  3, 5),
    _make_zone("LOWER_LEFT",  "Lower-left quadrant — third panel in a 2×2 layout",
               3, 4,  0, 2),
    _make_zone("LOWER_RIGHT", "Lower-right quadrant — fourth panel in a 2×2 layout",
               3, 4,  3, 5),

    # Absolute centre (for focus animations)
    _make_zone("CENTER",      "Absolute canvas centre — isolated focus element or equation",
               2, 3,  2, 3),
]}

ZONE_NAMES = list(ZONES.keys())


# ── Zone lookup by pixel position ─────────────────────────────────────────────

def zone_at_pixel(px: int, py: int) -> Zone:
    """Return the zone that contains pixel (px, py)."""
    col = min(int(px / PX_CELL_W), GRID_COLS - 1)
    row = min(int(py / PX_CELL_H), GRID_ROWS - 1)
    # Find smallest zone that contains this cell
    for zone in ZONES.values():
        if (zone.row_start <= row <= zone.row_end and
                zone.col_start <= col <= zone.col_end):
            return zone
    return ZONES["MAIN"]


# ── Zone → Manim code snippet ─────────────────────────────────────────────────

def zone_to_manim_position(zone_name: str) -> str:
    """
    Return a Manim positional snippet for the named zone center.
    Example: 'SIDEBAR' → '.move_to(np.array([4.67, 0.67, 0]))'
    The VLM Critic and code patcher inject this into regenerated code.
    """
    zone = ZONES.get(zone_name)
    if not zone:
        return ".move_to(ORIGIN)"
    x, y = zone.manim_center
    return f".move_to(np.array([{x}, {y}, 0]))"


def zone_to_shift_hint(from_zone: str, to_zone: str) -> str:
    """
    Return a natural-language shift instruction for the code patcher LLM.
    E.g.: 'Move the element from TITLE center (0.0, 3.33) to SIDEBAR center (4.67, 0.67)'
    """
    fz = ZONES.get(from_zone)
    tz = ZONES.get(to_zone)
    if not fz or not tz:
        return f"Move element to zone {to_zone}"
    return (
        f"Move the element from {from_zone} center {fz.manim_center} "
        f"to {to_zone} center {tz.manim_center} "
        f"({tz.description})"
    )


# ── Frame overlay (PIL) ───────────────────────────────────────────────────────

def draw_grid_overlay(image_bytes: bytes, alpha: int = 140) -> bytes:
    """
    Overlay the 6×6 grid + zone labels on a JPEG/PNG frame image.
    Returns PNG bytes of the annotated frame.

    alpha: grid line opacity (0=invisible, 255=solid)
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    # Resize to standard 1920×1080 if needed
    if img.size != (PX_W, PX_H):
        img = img.resize((PX_W, PX_H), Image.LANCZOS)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    # ── Grid lines ────────────────────────────────────────────────────────────
    line_colour = (255, 255, 0, alpha)
    for c in range(1, GRID_COLS):
        x = int(c * PX_CELL_W)
        draw.line([(x, 0), (x, PX_H)], fill=line_colour, width=2)
    for r in range(1, GRID_ROWS):
        y = int(r * PX_CELL_H)
        draw.line([(0, y), (PX_W, y)], fill=line_colour, width=2)

    # ── Cell coordinates in each cell corner (small grey text) ───────────────
    try:
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_zone  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font_small = ImageFont.load_default()
        font_zone  = font_small

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x0, y0 = _cell_topleft_px(r, c)
            draw.text(
                (x0 + 6, y0 + 4),
                f"({r},{c})",
                fill=(220, 220, 220, 200),
                font=font_small,
            )

    # ── Zone label in the centre of each major zone ───────────────────────────
    # Only label the 5 top-level zones to avoid clutter
    label_zones = ["TITLE", "MAIN", "SIDEBAR", "FOOTER", "CENTER"]
    for zname in label_zones:
        zone = ZONES[zname]
        bx0, by0, bx1, by1 = zone.pixel_bbox
        cx, cy = (bx0 + bx1) // 2, (by0 + by1) // 2
        # Semi-transparent background pill
        tw = len(zname) * 14 + 16
        th = 30
        draw.rectangle(
            [(cx - tw//2, cy - th//2), (cx + tw//2, cy + th//2)],
            fill=(30, 30, 30, 180),
        )
        draw.text(
            (cx - tw//2 + 8, cy - th//2 + 4),
            zname,
            fill=(80, 220, 255, 230),
            font=font_zone,
        )

    # ── Composite ─────────────────────────────────────────────────────────────
    composited = Image.alpha_composite(img, overlay).convert("RGB")
    buf = io.BytesIO()
    composited.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def frame_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes as base64 string for API submission."""
    return base64.b64encode(image_bytes).decode("utf-8")


# ── Zone manifest for VLM prompts ─────────────────────────────────────────────

def zone_manifest() -> str:
    """
    Return a compact text table of all zones for inclusion in the VLM prompt.
    Each row: ZONE_NAME | Manim center (x,y) | Description
    """
    lines = ["ZONE          | MANIM CENTER   | DESCRIPTION"]
    lines.append("-" * 70)
    for zone in ZONES.values():
        x, y = zone.manim_center
        lines.append(f"{zone.name:<14}| ({x:+.2f}, {y:+.2f})    | {zone.description}")
    return "\n".join(lines)