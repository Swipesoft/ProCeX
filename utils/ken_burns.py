"""
utils/ken_burns.py
Turn a static image into a cinematic timed video clip using FFmpeg's
zoompan filter — the Ken Burns effect.
Used for IMAGE_GEN scenes (NanoBanana output → scene clip).
"""
from __future__ import annotations
import subprocess
import os
from config import RESOLUTIONS


EFFECTS = {
    "zoom_in":    "z='min(zoom+0.0008,1.4)'",
    "zoom_out":   "z='if(lte(zoom,1.0),1.4,max(zoom-0.0008,1.0))'",
    "pan_right":  "z='1.2':x='iw/2-(iw/zoom/2)+(iw/zoom/4)*on/nd'",
    "pan_left":   "z='1.2':x='iw/2-(iw/zoom/2)-(iw/zoom/4)*on/nd'",
    "drift":      "z='min(zoom+0.0005,1.25)':x='iw/2-(iw/zoom/2)+sin(on/nd*PI)*50'",
}


def image_to_video_clip(
    image_path:  str,
    duration:    float,
    output_path: str,
    resolution:  str = "1080p",
    effect:      str = "zoom_in",
    fps:         int = 25,
) -> str:
    """
    Render a static image into a video clip with Ken Burns motion.
    Scales input image to safe dimensions before zoompan to prevent
    filter failures with arbitrary NanoBanana output sizes.
    """
    res      = RESOLUTIONS[resolution]
    n_frames = int(duration * fps)
    zp_expr  = EFFECTS.get(effect, EFFECTS["zoom_in"])

    # Scale input to 2× output size before zoompan — zoompan needs headroom
    # to zoom/pan without hitting the edge of the source image.
    # pad to even dimensions, force yuv420p throughout.
    scale_w = res.width  * 2
    scale_h = res.height * 2

    vf = (
        f"scale={scale_w}:{scale_h}:force_original_aspect_ratio=increase,"
        f"crop={scale_w}:{scale_h},"
        f"zoompan={zp_expr}:d={n_frames}:s={res.width}x{res.height}:fps={fps},"
        f"format=yuv420p"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", image_path,
        "-vf", vf,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=120,
        encoding="utf-8", errors="replace",
    )

    if result.returncode != 0:
        # zoompan failed — fall back to simple scale+fade (no motion)
        vf_simple = (
            f"scale={res.width}:{res.height}:force_original_aspect_ratio=decrease,"
            f"pad={res.width}:{res.height}:-1:-1:color=black,"
            f"format=yuv420p"
        )
        cmd_simple = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", image_path,
            "-vf", vf_simple,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        result2 = subprocess.run(
            cmd_simple, capture_output=True, text=True,
            timeout=120,
            encoding="utf-8", errors="replace",
        )
        if result2.returncode != 0:
            raise RuntimeError(
                f"Ken Burns FFmpeg failed:\n{result.stderr[-500:]}\n"
                f"Fallback also failed:\n{result2.stderr[-300:]}"
            )

    return output_path


def cycle_effect(scene_id: int) -> str:
    """Cycle through effects to avoid repetition across scenes."""
    effects = list(EFFECTS.keys())
    return effects[scene_id % len(effects)]