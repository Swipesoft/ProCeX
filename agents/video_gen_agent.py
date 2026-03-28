"""
agents/video_gen_agent.py

VideoGenAgent — generates short live-video clips (4-12s) using the
Novita Seedance 1.5 Pro API and integrates them into ProcEx scenes.

Two modes:
  T2V (text-to-video): generates from a text prompt alone
  I2V (image-to-video): animates an existing IMAGE_GEN scene image

Budget rule: max 16s of live video per 60s of total video duration.
  e.g. 6min video → max 96s live video → ~12 clips of 8s each
  VisualDirector flags scenes as VIDEO_GEN when appropriate.
  VideoGenAgent enforces the budget and downgrades excess to IMAGE_GEN.

API endpoints (async — returns task_id, poll for result):
  POST https://api.novita.ai/v3/async/seedance-v1.5-pro-t2v
  POST https://api.novita.ai/v3/async/seedance-v1.5-pro-i2v
  GET  https://api.novita.ai/v3/async/task-result/{task_id}

Key constraints from API docs:
  - Duration: 4-12s
  - Resolution: 480p or 720p (NOT 1080p)
  - Ratio: 9:16, 16:9, 1:1 etc. or "adaptive"
  - generate_audio must be False (we supply our own TTS audio)
  - watermark: False

Authentication: Bearer token via NOVITA_API_KEY in .env
"""
from __future__ import annotations

import base64
import os
import time
from typing import Optional

import requests

from agents.base_agent import BaseAgent
from config import ProcExConfig, RESOLUTIONS
from state import ProcExState, VisualStrategy
from utils.llm_client import LLMClient


# ── API constants ─────────────────────────────────────────────────────────────
NOVITA_T2V_URL     = "https://api.novita.ai/v3/async/seedance-v1.5-pro-t2v"
NOVITA_I2V_URL     = "https://api.novita.ai/v3/async/seedance-v1.5-pro-i2v"
NOVITA_RESULT_URL  = "https://api.novita.ai/v3/async/task-result"  # task_id as query param

POLL_INTERVAL_SECS  = 5
POLL_TIMEOUT_SECS   = 300   # 5 min max wait per clip
MAX_VIDEO_RETRIES   = 3     # submit retries on 429/503

# Budget: live video seconds per minute of total video
VIDEO_BUDGET_RATIO = 16 / 60   # 16s per 60s

# Clip duration: use 5s clips by default — long enough to be meaningful,
# short enough to stay within budget easily
DEFAULT_CLIP_SECS  = 5


class VideoGenAgent(BaseAgent):
    """
    Generates VIDEO_GEN scenes using Novita Seedance 1.5 Pro.
    Called from parallel_runner alongside ManimCoder and ImageGenAgent.
    """
    name = "VideoGenAgent"

    def __init__(self, cfg: ProcExConfig, llm: LLMClient):
        super().__init__(cfg, llm)
        self._api_key = os.environ.get("NOVITA_API_KEY", "")

    def run(self, state: ProcExState) -> ProcExState:
        """Not used directly — generate_for_scene() is called per-scene."""
        return state

    # ── Budget enforcement ────────────────────────────────────────────────────

    @staticmethod
    def compute_budget(state: ProcExState) -> float:
        """Return max total seconds of live video allowed for this run."""
        return state.target_duration_minutes * 60.0 * VIDEO_BUDGET_RATIO

    @staticmethod
    def enforce_budget(state: ProcExState) -> ProcExState:
        """
        Downgrade VIDEO_GEN scenes that exceed the budget to IMAGE_GEN.
        Called once before Stage B begins.
        Budget = 16s per 60s of total video.
        """
        budget_secs = VideoGenAgent.compute_budget(state)
        used        = 0.0
        downgraded  = 0

        for scene in state.scenes:
            if scene.visual_strategy != VisualStrategy.VIDEO_GEN:
                continue
            clip_dur = min(
                max(4.0, scene.duration_seconds),
                DEFAULT_CLIP_SECS,
            )
            if used + clip_dur <= budget_secs:
                used += clip_dur
            else:
                scene.visual_strategy = VisualStrategy.IMAGE_GEN
                downgraded += 1

        if downgraded:
            import logging
            logging.getLogger("VideoGenAgent").info(
                f"Budget enforcement: {downgraded} VIDEO_GEN scenes downgraded to "
                f"IMAGE_GEN (budget={budget_secs:.0f}s used={used:.0f}s)"
            )
        return state

    # ── Per-scene generation ──────────────────────────────────────────────────

    def generate_for_scene(
        self,
        scene,
        resolution: str,
        videos_dir: str,
    ) -> Optional[str]:
        """
        Generate a video clip for one VIDEO_GEN scene.
        Returns path to the downloaded .mp4, or None on failure.
        """
        if not self._api_key:
            self._log("NOVITA_API_KEY not set — skipping VIDEO_GEN")
            return None

        res = RESOLUTIONS.get(resolution, RESOLUTIONS["1080p"])

        # Map our resolution to Novita's (max 720p)
        novita_res  = "720p"   # Seedance 1.5 Pro max
        # Map our aspect ratio to Novita's ratio string
        novita_ratio = "9:16" if res.is_portrait else "16:9"

        tts_dur   = getattr(scene, "tts_duration", 0.0) or scene.duration_seconds

        # API max is 12s but >8s is expensive and the quality gap is minimal.
        # For scenes longer than 8s, generate an 8s clip and let the assembler
        # freeze-extend the last frame for the remainder. This is better than
        # the alternative (generating a static image for the whole duration)
        # because we still get 8s of live motion at the start.
        # For scenes <= 8s, match the actual TTS duration exactly (clamped 4-8).
        if tts_dur > 8.0:
            clip_secs = 8   # cap at 8s — remainder covered by freeze-extend
            self._log(
                f"Scene {scene.id}: tts_duration={tts_dur:.1f}s > 8s — "
                f"generating 8s clip, assembler will freeze-extend remaining "
                f"{tts_dur-8:.1f}s"
            )
        else:
            clip_secs = max(4, min(int(tts_dur), 8))

        # Choose T2V or I2V
        has_image = bool(scene.image_paths)
        if has_image:
            return self._i2v(scene, novita_ratio, novita_res, clip_secs, videos_dir)
        else:
            return self._t2v(scene, novita_ratio, novita_res, clip_secs, videos_dir)

    def _t2v(self, scene, ratio, resolution, duration, out_dir) -> Optional[str]:
        """Text-to-video generation."""
        prompt = self._build_prompt(scene)
        self._log(f"Scene {scene.id}: T2V — {duration}s, {ratio}, prompt={prompt[:80]}")

        payload = {
            "prompt":          prompt,
            "duration":        duration,
            "ratio":           ratio,
            "resolution":      resolution,
            "fps":             24,
            "watermark":       False,
            "generate_audio":  False,    # we supply our own TTS audio
            "camera_fixed":    False,
            "seed":            -1,
        }
        return self._submit_and_poll(payload, NOVITA_T2V_URL, scene, out_dir)

    def _i2v(self, scene, ratio, resolution, duration, out_dir) -> Optional[str]:
        """Image-to-video: animate the scene's existing generated image."""
        image_path = scene.image_paths[0]
        self._log(
            f"Scene {scene.id}: I2V — {duration}s, {ratio}, "
            f"image={os.path.basename(image_path)}"
        )

        # Encode image as base64
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            ext       = os.path.splitext(image_path)[1].lower().lstrip(".")
            mime      = f"image/{ext}" if ext in ("jpg","jpeg","png","webp") else "image/png"
            img_b64   = f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"
        except Exception as e:
            self._log(f"Scene {scene.id}: I2V image encode failed: {e} — falling back to T2V")
            return self._t2v(scene, ratio, resolution, duration, out_dir)

        prompt = self._build_prompt(scene)
        payload = {
            "image":           img_b64,
            "prompt":          prompt,
            "duration":        duration,
            "ratio":           "adaptive",  # use image's native ratio
            "resolution":      resolution,
            "fps":             24,
            "watermark":       False,
            "generate_audio":  False,
            "camera_fixed":    False,
            "seed":            -1,
        }
        return self._submit_and_poll(payload, NOVITA_I2V_URL, scene, out_dir)

    def _build_prompt(self, scene) -> str:
        """Build a Seedance-optimised video prompt from the scene."""
        base = scene.visual_prompt.strip()[:450]  # API recommends ≤500 chars

        # Add cinematic motion instruction — Seedance responds well to motion cues
        motion_hint = (
            "Slow cinematic camera drift. "
            "Atmospheric lighting. "
            "No text overlays. "
            "No watermarks. "
            "No UI elements."
        )
        return f"{base}. {motion_hint}"

    def _submit_and_poll(
        self,
        payload:  dict,
        url:      str,
        scene,
        out_dir:  str,
    ) -> Optional[str]:
        """Submit async task, poll for completion, download result."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        }

        # Submit — retry on rate limit (429) or server error (503)
        task_id    = None
        last_error = None

        for attempt in range(1, MAX_VIDEO_RETRIES + 1):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=30)

                # Parse retry-after from 429 response header or body
                if resp.status_code == 429:
                    retry_after = float(
                        resp.headers.get("Retry-After", "")
                        or resp.headers.get("X-RateLimit-Reset-After", "")
                        or "30"
                    )
                    import re as _re
                    body_match = _re.search(
                        r'retry.{0,10}?([\d.]+)\s*s',
                        resp.text, _re.IGNORECASE
                    )
                    if body_match:
                        retry_after = float(body_match.group(1)) + 2.0
                    self._log(
                        f"Scene {scene.id}: VIDEO_GEN 429 rate limit "
                        f"(attempt {attempt}/{MAX_VIDEO_RETRIES}) "
                        f"— retrying in {retry_after:.0f}s"
                    )
                    time.sleep(retry_after)
                    continue

                resp.raise_for_status()
                task_id = resp.json().get("task_id")
                if not task_id:
                    self._log(
                        f"Scene {scene.id}: no task_id in response: {resp.text[:200]}"
                    )
                    return None
                break   # success

            except requests.HTTPError as e:
                last_error = e
                status = getattr(e.response, "status_code", 0)
                if status in (503, 502) and attempt < MAX_VIDEO_RETRIES:
                    sleep = 10.0 * attempt
                    self._log(
                        f"Scene {scene.id}: VIDEO_GEN {status} "
                        f"(attempt {attempt}/{MAX_VIDEO_RETRIES}) — retrying in {sleep:.0f}s"
                    )
                    time.sleep(sleep)
                else:
                    self._log(f"Scene {scene.id}: VIDEO_GEN submit failed: {e}")
                    return None
            except Exception as e:
                last_error = e
                self._log(f"Scene {scene.id}: VIDEO_GEN submit error: {e}")
                return None

        if not task_id:
            self._log(
                f"Scene {scene.id}: VIDEO_GEN submit exhausted "
                f"{MAX_VIDEO_RETRIES} attempts — degrading to IMAGE_GEN"
            )
            return None

        scene.novita_task_id = task_id
        self._log(f"Scene {scene.id}: task submitted — task_id={task_id}")

        # Poll
        poll_url  = NOVITA_RESULT_URL   # task_id passed as query param below
        deadline  = time.time() + POLL_TIMEOUT_SECS
        while time.time() < deadline:
            time.sleep(POLL_INTERVAL_SECS)
            try:
                r = requests.get(
                    poll_url, headers=headers,
                    params={"task_id": task_id},  # query param not path
                    timeout=15
                )
                r.raise_for_status()
                data   = r.json()
                status = data.get("task", {}).get("status", "")

                if status == "TASK_STATUS_SUCCEED":
                    # API returns: {"videos": [{"video_url": "..."}], "task": {...}}
                    videos    = data.get("videos", [])
                    video_url = videos[0].get("video_url", "") if videos else ""
                    if not video_url:
                        self._log(f"Scene {scene.id}: task succeeded but no video URL")
                        return None
                    return self._download_clip(video_url, scene, out_dir, headers)

                elif status in ("TASK_STATUS_FAILED", "TASK_STATUS_EXPIRED"):
                    self._log(
                        f"Scene {scene.id}: VIDEO_GEN task {status}: "
                        f"{data.get('task', {}).get('err_message', '')}"
                    )
                    return None

                else:
                    self._log(f"Scene {scene.id}: polling... status={status}")

            except Exception as e:
                self._log(f"Scene {scene.id}: poll error: {e}")

        self._log(f"Scene {scene.id}: VIDEO_GEN timed out after {POLL_TIMEOUT_SECS}s")
        return None

    def _download_clip(
        self,
        url:     str,
        scene,
        out_dir: str,
        headers: dict,
    ) -> Optional[str]:
        """Download the generated video clip."""
        out_path = os.path.join(out_dir, f"scene_{scene.id:02d}_video.mp4")
        try:
            # S3 pre-signed URLs embed auth in query params — sending the
            # Novita Authorization header causes a 400 signature mismatch.
            download_headers = {}  # no auth header for S3
            r = requests.get(url, headers=download_headers, timeout=120, stream=True)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            self._log(f"Scene {scene.id}: video downloaded → {out_path}")
            return out_path
        except Exception as e:
            self._log(f"Scene {scene.id}: video download failed: {e}")
            return None