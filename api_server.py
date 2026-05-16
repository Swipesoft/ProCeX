"""
api_server.py — ProcEx FastAPI server (RunPod Serverless edition)
Branch: gemma-mode

Replaces Celery task dispatch with direct RunPod Serverless API calls.
RunPod manages its own job queue internally — no Redis broker needed.

Endpoints:
    POST /generate          — submit job to RunPod Serverless
    GET  /status/{job_id}   — poll RunPod job status
    GET  /video/{job_id}    — get pre-signed S3 URL for completed video
    GET  /health            — Railway health check
"""

import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── RunPod config ─────────────────────────────────────────────────────────────
RUNPOD_API_KEY     = os.environ["RUNPOD_API_KEY"]
RUNPOD_ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
RUNPOD_BASE_URL    = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type":  "application/json",
}

# ── App ───────────────────────────────────────────────────────────────────────
api = FastAPI(title="ProcEx API", version="2.0.0")

api.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("NEXTJS_URL", "*")],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Request schema ────────────────────────────────────────────────────────────
class VideoRequest(BaseModel):
    topic:      str
    mode:       str   = Field("research")
    provider:   str   = Field("gemma")
    minutes:    float = Field(4.0)
    style:      str   = Field("youtube-tutorial")
    resolution: str   = Field("1080p")
    context:    str   = Field("")


# ── Routes ────────────────────────────────────────────────────────────────────
@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/generate")
async def generate(req: VideoRequest):
    """Submit a video generation job to RunPod Serverless."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{RUNPOD_BASE_URL}/run",
            headers=HEADERS,
            json={"input": req.dict()},
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"RunPod error: {response.text}"
        )

    data   = response.json()
    job_id = data.get("id")

    return {"job_id": job_id, "status": "pending"}


@api.get("/status/{job_id}")
async def status(job_id: str):
    """Poll RunPod Serverless for job status."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{RUNPOD_BASE_URL}/status/{job_id}",
            headers=HEADERS,
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"RunPod error: {response.text}"
        )

    data   = response.json()
    status = data.get("status", "").upper()

    # RunPod statuses: IN_QUEUE → IN_PROGRESS → COMPLETED | FAILED
    if status == "IN_QUEUE":
        return {"job_id": job_id, "status": "pending"}

    elif status == "IN_PROGRESS":
        return {"job_id": job_id, "status": "running",
                "progress": "Pipeline is running..."}

    elif status == "COMPLETED":
        output = data.get("output", {})
        return {
            "job_id":        job_id,
            "status":        "done",
            "video_url":     output.get("video_url"),
            "subtitles_url": output.get("subtitles_url"),
        }

    elif status == "FAILED":
        return {
            "job_id": job_id,
            "status": "failed",
            "error":  data.get("error", "Unknown error"),
        }

    return {"job_id": job_id, "status": status.lower()}


@api.get("/video/{job_id}")
async def video(job_id: str):
    """Return pre-signed video URL for a completed job."""
    result = await status(job_id)
    if result["status"] != "done":
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not complete (status: {result['status']})"
        )
    return {
        "video_url":     result.get("video_url"),
        "subtitles_url": result.get("subtitles_url"),
    }