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

    Args:
        image_path:  path to PNG/JPG from NanoBanana
        duration:    clip length in seconds (matches scene TTS duration)
        output_path: output .mp4 path
        resolution:  "720p" | "1080p" | "4K"
        effect:      one of EFFECTS keys
        fps:         frames per second

    Returns:
        output_path on success
    """
    res      = RESOLUTIONS[resolution]
    n_frames = int(duration * fps)
    zp_expr  = EFFECTS.get(effect, EFFECTS["zoom_in"])

    # Full zoompan expression: effect + dimensions + duration
    zoompan = (
        f"zoompan={zp_expr}"
        f":d={n_frames}"
        f":s={res.width}x{res.height}"
        f":fps={fps}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", image_path,
        "-vf", f"{zoompan},format=yuv420p",
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        raise RuntimeError(f"Ken Burns FFmpeg failed:\n{result.stderr}")

    return output_path


def cycle_effect(scene_id: int) -> str:
    """Cycle through effects to avoid repetition across scenes."""
    effects = list(EFFECTS.keys())
    return effects[scene_id % len(effects)]
