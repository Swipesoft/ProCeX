"""
utils/label_overlay.py
Add callout labels to NanoBanana-generated anatomical images using Pillow.
Called when scene.needs_labels = True and scene.label_list is populated.
"""
from __future__ import annotations
import math
from pathlib import Path


def add_labels(
    image_path:  str,
    labels:      list[str],
    output_path: str | None = None,
) -> str:
    """
    Add styled callout labels arranged around the image border.

    Args:
        image_path:  path to source PNG from NanoBanana
        labels:      list of label strings (e.g. ["glomerulus", "Bowman's capsule"])
        output_path: destination path; defaults to <image_path>_labeled.png

    Returns:
        path to labeled image
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError("Pillow required: pip install Pillow")

    if output_path is None:
        p = Path(image_path)
        output_path = str(p.parent / f"{p.stem}_labeled{p.suffix}")

    img  = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    W, H = img.size

    # Try to load a clean sans-serif font — cross-platform search
    font_size = max(18, W // 60)
    font_candidates = [
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    font_bold_candidates = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]

    font = font_bold = None
    for path in font_candidates:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except Exception:
            continue
    for path in font_bold_candidates:
        try:
            font_bold = ImageFont.truetype(path, font_size)
            break
        except Exception:
            continue

    if font is None:
        font = ImageFont.load_default()
    if font_bold is None:
        font_bold = font

    n = len(labels)
    if n == 0:
        img.save(output_path)
        return output_path

    # Distribute labels around the perimeter
    # Place them at equal angular intervals, pushed to the image edge
    margin = int(W * 0.04)
    line_color  = "#00D4FF"   # cyan — matches cinematic palette
    text_color  = "#F0F0FF"
    bg_color    = (10, 10, 20, 200)  # dark semi-transparent

    for i, label in enumerate(labels):
        angle = (2 * math.pi * i / n) - math.pi / 2  # start from top

        # Anchor point on a slightly inset ellipse
        cx = W // 2 + int((W * 0.38) * math.cos(angle))
        cy = H // 2 + int((H * 0.38) * math.sin(angle))

        # Label text box position (further out)
        tx = W // 2 + int((W * 0.46) * math.cos(angle))
        ty = H // 2 + int((H * 0.46) * math.sin(angle))
        tx = max(margin, min(W - margin * 6, tx))
        ty = max(margin, min(H - margin * 2, ty))

        # Draw connecting line from anchor to label
        draw.line([(cx, cy), (tx, ty)], fill=line_color, width=max(2, W // 800))

        # Measure text
        bbox = draw.textbbox((0, 0), label, font=font_bold)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pad = 6

        # Draw background pill
        draw.rounded_rectangle(
            [tx - pad, ty - pad, tx + tw + pad, ty + th + pad],
            radius=4,
            fill=bg_color,
            outline=line_color,
            width=1,
        )

        # Draw text
        draw.text((tx, ty), label, font=font_bold, fill=text_color)

        # Dot at anchor
        r = max(4, W // 300)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=line_color)

    img.save(output_path)
    return output_path
