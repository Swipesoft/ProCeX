"""
tasks.py — ProcEx Celery task definitions
Branch: gemma-mode

Each task runs ONE full ProcEx pipeline as a subprocess, then uploads
the output video + subtitles to S3 and returns pre-signed URLs.

Default resolution is 1080p_v (matching the gemma-mode CLI convention).
"""

import os
import json
import subprocess
import boto3

from pathlib import Path
from celery_app import app
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

# ── S3 client (initialised once per worker process) ──────────────────────────
s3 = boto3.client(
    "s3",
    aws_access_key_id     = os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name           = os.environ["AWS_REGION"],
)
BUCKET = os.environ["S3_BUCKET"]


def _presign(key: str, expires: int = 86400) -> str:
    """Generate a 24-hour pre-signed GET URL for an S3 object."""
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": key},
        ExpiresIn=expires,
    )


@app.task(bind=True, name="procex.generate_video", max_retries=1)
def generate_video(self, job_id: str, params: dict) -> dict:
    """
    Run the full ProcEx pipeline and upload result to S3.

    Params dict mirrors the VideoRequest model in api_server.py:
        topic       str   — video topic
        mode        str   — "research" | "pdf"
        provider    str   — "gemma" (Modal endpoint)
        minutes     float — target video length
        style       str   — "youtube-tutorial" etc.
        resolution  str   — "1080p_v" (gemma-mode default)
        context     str   — optional system context string

    Returns:
        {"video_url": str, "subtitles_url": str | None}
    """
    output_dir = f"/tmp/procex_output/{job_id}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Build CLI command ─────────────────────────────────────────────────────
    cmd = [
        "python", "main.py",
        "--topic",      params["topic"],
        "--mode",       params.get("mode",       "research"),
        "--provider",   params.get("provider",   "gemma"),
        "--minutes",    str(params.get("minutes", 4)),
        "--style",      params.get("style",      "youtube-tutorial"),
        "--resolution", params.get("resolution", "1080p"),
        "--output-dir", output_dir,
    ]

    # Context is optional but can be very long — pass as a single argument
    context = params.get("context", "").strip()
    if context:
        cmd += ["--context", context]

    logger.info("[%s] Starting pipeline: %s", job_id, params.get("topic"))
    self.update_state(state="RUNNING", meta={"progress": "Pipeline started..."})

    # ── Run pipeline ──────────────────────────────────────────────────────────
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd="/app",                    # repo root inside the container
    )

    if result.returncode != 0:
        logger.error("[%s] Pipeline stderr:\n%s", job_id, result.stderr[-3000:])
        raise Exception(f"Pipeline failed (exit {result.returncode}): "
                        f"{result.stderr[-2000:]}")

    logger.info("[%s] Pipeline finished. Locating output video...", job_id)

    # ── Find the output video ─────────────────────────────────────────────────
    # main.py prints "Video saved to: /path/to/video.mp4" on success
    video_path: Path | None = None
    for line in result.stdout.splitlines():
        if "Video saved to:" in line:
            video_path = Path(line.split(":", 1)[-1].strip())
            break

    # Fallback: scan output dir for any .mp4
    if video_path is None or not video_path.exists():
        mp4s = list(Path(output_dir).rglob("*.mp4"))
        if mp4s:
            video_path = mp4s[0]
        else:
            raise Exception(
                f"No video file found after pipeline run. "
                f"stdout tail: {result.stdout[-1000:]}"
            )

    logger.info("[%s] Found video at %s", job_id, video_path)
    self.update_state(state="RUNNING", meta={"progress": "Uploading to S3..."})

    # ── Upload video to S3 ────────────────────────────────────────────────────
    s3_video_key = f"{job_id}/video.mp4"
    s3.upload_file(
        str(video_path),
        BUCKET,
        s3_video_key,
        ExtraArgs={"ContentType": "video/mp4"},
    )
    logger.info("[%s] Uploaded video → s3://%s/%s", job_id, BUCKET, s3_video_key)

    # ── Upload subtitles if present ───────────────────────────────────────────
    srt_url: str | None = None
    srt_path = video_path.with_suffix(".srt")
    if srt_path.exists():
        s3_srt_key = f"{job_id}/subtitles.srt"
        s3.upload_file(str(srt_path), BUCKET, s3_srt_key)
        srt_url = _presign(s3_srt_key)
        logger.info("[%s] Uploaded subtitles → s3://%s/%s", job_id, BUCKET, s3_srt_key)

    # ── Save metadata ─────────────────────────────────────────────────────────
    metadata = {**params, "job_id": job_id}
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{job_id}/metadata.json",
        Body=json.dumps(metadata, indent=2),
        ContentType="application/json",
    )

    video_url = _presign(s3_video_key)

    logger.info("[%s] Done. video_url=%s...", job_id, video_url[:60])
    return {"video_url": video_url, "subtitles_url": srt_url}