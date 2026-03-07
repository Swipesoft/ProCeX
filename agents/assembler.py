"""
agents/assembler.py
Final stage: concatenate all scene clips, overlay TTS audio, add subtitles.
Output: output/videos/<topic_slug>.mp4
"""
from __future__ import annotations
import os
import platform
import subprocess
import tempfile
from state import ProcExState
from config import ProcExConfig, RESOLUTIONS
from utils.llm_client import LLMClient
from utils.timestamp_utils import timestamps_to_dict_list
from agents.base_agent import BaseAgent


class AssemblerAgent(BaseAgent):
    name = "AssemblerAgent"

    def run(self, state: ProcExState) -> ProcExState:
        videos_dir = self.cfg.dirs["videos"]
        os.makedirs(videos_dir, exist_ok=True)

        clips = [
            s.clip_path for s in state.scenes
            if s.clip_path and os.path.exists(s.clip_path)
                           and os.path.getsize(s.clip_path) > 0
        ]

        if not clips:
            self._err(state, "No clips to assemble")
            return state

        self._log(f"Assembling {len(clips)} clips + audio...")

        res          = RESOLUTIONS[state.resolution]
        silent_video = os.path.join(videos_dir, f"{state.topic_slug}_silent.mp4")
        final_video  = os.path.join(videos_dir, f"{state.topic_slug}.mp4")

        # ── Step 1: Normalize all clips ───────────────────────────────────
        normalized = self._normalize_clips(clips, res, videos_dir)

        # ── Step 2: Concatenate ───────────────────────────────────────────
        self._concat_clips(normalized, silent_video)
        self._log(f"Concatenated -> {silent_video}")

        # ── Step 3: Overlay audio ─────────────────────────────────────────
        if state.audio_path and os.path.exists(state.audio_path):
            self._overlay_audio(silent_video, state.audio_path, final_video)
            self._log(f"Audio overlaid -> {final_video}")
        else:
            self._log("WARNING: No audio — outputting silent video")
            import shutil
            shutil.copy2(silent_video, final_video)

        # ── Step 4: Generate SRT subtitles ────────────────────────────────
        srt_path = os.path.join(videos_dir, f"{state.topic_slug}.srt")
        try:
            self._generate_srt(state, srt_path)
            self._log(f"Subtitles -> {srt_path}")
        except Exception as e:
            self._log(f"SRT generation failed (non-critical): {e}")

        # ── Step 5: Burn subtitles ─────────────────────────────────────────
        # Skipped on Windows — libass requires font cache config not present
        # by default. The .srt file is still usable in any video player.
        if platform.system() != "Windows" and os.path.exists(srt_path):
            cc_video = os.path.join(videos_dir, f"{state.topic_slug}_cc.mp4")
            try:
                self._burn_subtitles(final_video, srt_path, cc_video)
                self._log(f"Captioned video -> {cc_video}")
            except Exception as e:
                self._log(f"Subtitle burn failed (non-critical): {e}")
        elif platform.system() == "Windows":
            self._log(
                "Subtitle burn skipped on Windows (libass not configured). "
                f"Load the .srt file manually in VLC: "
                f"Subtitle -> Add Subtitle File -> {srt_path}"
            )

        state.final_video_path = final_video
        self._log(f"Final video: {final_video}")

        size_mb  = os.path.getsize(final_video) / 1_048_576
        duration = self._probe_duration(final_video)
        self._log(
            f"Video stats: {duration:.1f}s ({duration/60:.1f} min), {size_mb:.1f} MB"
        )

        return state

    # ── Normalize ─────────────────────────────────────────────────────────────

    def _normalize_clips(self, clips: list[str], res, out_dir: str) -> list[str]:
        """
        Normalize each clip to the same resolution, codec, and framerate.

        Strategy:
          1. Probe the clip with ffprobe
          2. If already correct (target WxH, h264, ~25fps, yuv420p) — stream copy
          3. Otherwise re-encode with explicit scale to exact target dimensions
             (no force_original_aspect_ratio — just hard scale, add padding if needed)
        """
        normalized = []
        target_w, target_h = res.width, res.height

        for i, clip in enumerate(clips):
            norm_path = os.path.join(out_dir, f"norm_{i:02d}.mp4")

            # Probe clip properties
            info = self._probe_clip(clip)

            # Check if normalization is actually needed
            if (
                info.get("width")  == target_w
                and info.get("height") == target_h
                and info.get("codec")  == "h264"
                and abs(info.get("fps", 0) - 25) < 1
                and info.get("pix_fmt", "") in ("yuv420p", "yuvj420p")
            ):
                # Already correct — stream copy (fast, lossless)
                import shutil
                shutil.copy2(clip, norm_path)
                normalized.append(norm_path)
                self._log(f"Clip {i}: already correct format — copied")
                continue

            # Needs re-encoding — scale to exact target, pad if aspect differs
            cmd = [
                "ffmpeg", "-y",
                "-i", clip,
                "-vf", (
                    f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                    f"pad={target_w}:{target_h}:-1:-1:color=black,"
                    f"fps=25"
                ),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                norm_path,
            ]

            # Timeout scales with resolution and duration
            duration = info.get("duration", 30)
            # At 4K fast preset, encoding speed is ~1-3x realtime
            timeout = max(120, int(duration * 8))

            result = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=timeout,
            )

            # Check output file is non-empty
            if result.returncode == 0 and os.path.exists(norm_path) and os.path.getsize(norm_path) > 0:
                normalized.append(norm_path)
                self._log(f"Clip {i}: normalized ({info.get('width')}x{info.get('height')} -> {target_w}x{target_h})")
            else:
                # Log the actual error (last 300 chars of stderr, skipping the
                # truncated leading "ut file is empty..." prefix)
                err = result.stderr.strip()[-300:] if result.stderr else "unknown error"
                self._log(f"Clip {i}: normalization failed, using original. Error: {err}")
                normalized.append(clip)

        return normalized

    def _probe_clip(self, clip: str) -> dict:
        """Probe a clip's properties using ffprobe. Returns a dict with keys:
        width, height, codec, fps, pix_fmt, duration."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,codec_name,r_frame_rate,pix_fmt"
            ":format=duration",
            "-of", "default=noprint_wrappers=1",
            clip,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=15,
            )
            info: dict = {}
            for line in result.stdout.splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip()
                    if k == "width":             info["width"]   = int(v)
                    elif k == "height":          info["height"]  = int(v)
                    elif k == "codec_name":      info["codec"]   = v
                    elif k == "pix_fmt":         info["pix_fmt"] = v
                    elif k == "duration":
                        try: info["duration"] = float(v)
                        except ValueError: pass
                    elif k == "r_frame_rate" and "/" in v:
                        num, den = v.split("/")
                        try: info["fps"] = int(num) / max(int(den), 1)
                        except (ValueError, ZeroDivisionError): pass
            return info
        except Exception:
            return {}

    # ── Concat ────────────────────────────────────────────────────────────────

    def _concat_clips(self, clips: list[str], output: str) -> None:
        """Concatenate clips using FFmpeg concat demuxer.
        Forward slashes required in the list file, even on Windows."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            list_path = f.name
            for clip in clips:
                safe = os.path.abspath(clip).replace("\\", "/")
                f.write(f"file '{safe}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            output,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=600,
        )
        os.unlink(list_path)

        if result.returncode != 0:
            raise RuntimeError(f"Concat failed:\n{result.stderr[-2000:]}")

    # ── Audio overlay ─────────────────────────────────────────────────────────

    def _overlay_audio(self, video: str, audio: str, output: str) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-i", video, "-i", audio,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            output,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Audio overlay failed:\n{result.stderr[-1000:]}")

    # ── SRT ───────────────────────────────────────────────────────────────────

    def _generate_srt(self, state: ProcExState, srt_path: str) -> None:
        if not state.all_timestamps:
            return

        def srt_time(s: float) -> str:
            h  = int(s // 3600)
            m  = int((s % 3600) // 60)
            sc = int(s % 60)
            ms = int((s % 1) * 1000)
            return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"

        WORDS_PER_SUB = 7
        chunks = []
        for i in range(0, len(state.all_timestamps), WORDS_PER_SUB):
            chunk = state.all_timestamps[i : i + WORDS_PER_SUB]
            if chunk:
                chunks.append((
                    chunk[0].start,
                    chunk[-1].end,
                    " ".join(t.word for t in chunk),
                ))

        with open(srt_path, "w", encoding="utf-8") as f:
            for idx, (start, end, text) in enumerate(chunks, 1):
                f.write(f"{idx}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n\n")

    def _burn_subtitles(self, video: str, srt: str, output: str) -> None:
        """Burn subtitles using libass. Only called on non-Windows."""
        srt_escaped = srt.replace("\\", "/").replace(":", "\\:")
        cmd = [
            "ffmpeg", "-y", "-i", video,
            "-vf", (
                f"subtitles={srt_escaped}:"
                f"force_style='FontName=Arial,FontSize=22,"
                f"PrimaryColour=&H00F0F0FF,OutlineColour=&H000A0A0F,"
                f"Outline=2,Alignment=2,MarginV=40'"
            ),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            output,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Subtitle burn failed: {result.stderr[-500:]}")

    # ── Probe ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _probe_duration(video: str) -> float:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0