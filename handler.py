"""
handler.py — RunPod Serverless entry point
Branch: gemma-mode

This replaces tasks.py + celery_app.py entirely.
RunPod calls runpod.serverless.start() which invokes handler()
for each incoming job.

Input payload mirrors the VideoRequest model in api_server.py:
    {
        "topic":      str,
        "mode":       str,   # "research" | "pdf"
        "provider":   str,   # "gemma"
        "minutes":    float,
        "style":      str,
        "resolution": str,   # "1080p"
        "context":    str    # optional
    }

Output:
    {
        "video_url":     str,   # 24h pre-signed S3 GET URL
        "subtitles_url": str | None
    }
"""

import os
import json
import uuid
import subprocess
import boto3
import runpod

from pathlib import Path

# ── S3 client ─────────────────────────────────────────────────────────────────
s3 = boto3.client(
    "s3",
    aws_access_key_id     = os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name           = os.environ["AWS_REGION"],
)
BUCKET = os.environ["S3_BUCKET"]


def _presign(key: str, expires: int = 86400) -> str:
    """Generate a 24h pre-signed GET URL."""
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": key},
        ExpiresIn=expires,
    )


def handler(job: dict) -> dict:
    """
    RunPod calls this function for every incoming job.

    job = {
        "id":    str,   # RunPod job ID
        "input": dict   # VideoRequest params
    }
    """
    params  = job["input"]
    job_id  = job.get("id", str(uuid.uuid4()))

    print(f"[{job_id}] Starting pipeline: {params.get('topic')}")

    # ── Output directory ──────────────────────────────────────────────────────
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

    context = params.get("context", "").strip()
    if context:
        cmd += ["--context", context]

    # ── Run pipeline ──────────────────────────────────────────────────────────
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd="/app",
    )

    if result.returncode != 0:
        error_tail = result.stderr[-2000:]
        print(f"[{job_id}] Pipeline failed:\n{error_tail}")
        raise Exception(f"Pipeline failed (exit {result.returncode}): {error_tail}")

    print(f"[{job_id}] Pipeline complete. Locating output video...")

    # ── Find output video ─────────────────────────────────────────────────────
    video_path = None
    for line in result.stdout.splitlines():
        if "Video saved to:" in line:
            video_path = Path(line.split(":", 1)[-1].strip())
            break

    if video_path is None or not video_path.exists():
        mp4s = list(Path(output_dir).rglob("*.mp4"))
        if mp4s:
            video_path = mp4s[0]
        else:
            raise Exception(
                f"No video file found. stdout tail:\n{result.stdout[-1000:]}"
            )

    print(f"[{job_id}] Found video: {video_path}")

    # ── Upload video to S3 ────────────────────────────────────────────────────
    s3_video_key = f"{job_id}/video.mp4"
    s3.upload_file(
        str(video_path),
        BUCKET,
        s3_video_key,
        ExtraArgs={"ContentType": "video/mp4"},
    )
    print(f"[{job_id}] Uploaded → s3://{BUCKET}/{s3_video_key}")

    # ── Upload subtitles if present ───────────────────────────────────────────
    srt_url = None
    srt_path = video_path.with_suffix(".srt")
    if srt_path.exists():
        s3_srt_key = f"{job_id}/subtitles.srt"
        s3.upload_file(str(srt_path), BUCKET, s3_srt_key)
        srt_url = _presign(s3_srt_key)
        print(f"[{job_id}] Uploaded subtitles → s3://{BUCKET}/{s3_srt_key}")

    # ── Save metadata ─────────────────────────────────────────────────────────
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{job_id}/metadata.json",
        Body=json.dumps({**params, "job_id": job_id}, indent=2),
        ContentType="application/json",
    )

    video_url = _presign(s3_video_key)
    print(f"[{job_id}] Done.")

    return {
        "video_url":     video_url,
        "subtitles_url": srt_url,
    }


# ── RunPod serverless entry point ─────────────────────────────────────────────
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})