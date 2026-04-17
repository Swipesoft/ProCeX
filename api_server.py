"""
api_server.py — ProcEx FastAPI server
Branch: gemma-mode

Endpoints:
    POST /generate          — enqueue a video generation job
    GET  /status/{job_id}   — poll job status
    GET  /video/{job_id}    — get pre-signed S3 URL for completed video
    GET  /health            — Railway health check
"""

import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from celery.result import AsyncResult
from celery_app import app as celery_app
from tasks import generate_video

# ── App ───────────────────────────────────────────────────────────────────────
api = FastAPI(title="ProcEx API", version="1.0.0")

api.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("NEXTJS_URL", "*")],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Request schema ────────────────────────────────────────────────────────────
class VideoRequest(BaseModel):
    topic:      str
    mode:       str   = Field("research",        description="research | pdf")
    provider:   str   = Field("gemma",           description="LLM provider")
    minutes:    float = Field(4.0,               description="Target video length")
    style:      str   = Field("youtube-tutorial",description="Video style")
    resolution: str   = Field("1080p",           description="Output resolution")
    context:    str   = Field("",                description="Optional system context")


# ── Routes ────────────────────────────────────────────────────────────────────
@api.get("/health")
def health():
    """Railway health check endpoint."""
    return {"status": "ok"}


@api.post("/generate")
def generate(req: VideoRequest):
    """Enqueue a video generation job and return a job_id to poll."""
    job_id = str(uuid.uuid4())
    generate_video.apply_async(
        kwargs={"job_id": job_id, "params": req.dict()},
        task_id=job_id,
        queue="procex",
    )
    return {"job_id": job_id, "status": "pending"}


@api.get("/status/{job_id}")
def status(job_id: str):
    """Poll job status. States: pending → running → done | failed"""
    result = AsyncResult(job_id, app=celery_app)

    if result.state == "PENDING":
        return {"job_id": job_id, "status": "pending"}

    elif result.state == "STARTED":
        return {"job_id": job_id, "status": "running", "progress": "Starting..."}

    elif result.state == "RUNNING":
        meta = result.info or {}
        return {
            "job_id":   job_id,
            "status":   "running",
            "progress": meta.get("progress", ""),
        }

    elif result.state == "SUCCESS":
        return {"job_id": job_id, "status": "done", **result.result}

    elif result.state == "FAILURE":
        return {
            "job_id": job_id,
            "status": "failed",
            "error":  str(result.result),
        }

    return {"job_id": job_id, "status": result.state.lower()}


@api.get("/video/{job_id}")
def video(job_id: str):
    """Return pre-signed video URL for a completed job."""
    result = AsyncResult(job_id, app=celery_app)
    if result.state != "SUCCESS":
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} is not complete (state: {result.state})"
        )
    return result.result   # {"video_url": ..., "subtitles_url": ...}