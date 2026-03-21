"""
agents/assembler.py
Final stage: sync each scene clip to its TTS audio, then concatenate.

Sync strategy (segment-level, not global):
  For each scene:
    1. Probe rendered clip duration and TTS audio chunk duration
    2. If clip SHORT (Manim didn't fill narration time):
         freeze-extend the last frame with tpad=stop_mode=clone
    3. If clip LONG (Manim overran — rare):
         trim clip to audio duration with -t
    4. Mux synced video + scene audio into one segment
  Then concatenate all segments into the final video.

This is immune to the "animations N scenes ahead" drift caused by
Manim code not summing to scene.duration_seconds. Each scene's video
is stretched/trimmed to match its own audio before joining.

Fallback: if scene has no tts_audio_path, legacy concat-then-overlay.
"""
from __future__ import annotations
import os
import platform
import shutil
import subprocess
import tempfile
from state import ProcExState, Scene
from config import ProcExConfig, RESOLUTIONS
from utils.llm_client import LLMClient
from agents.base_agent import BaseAgent


class AssemblerAgent(BaseAgent):
    name = "AssemblerAgent"

    def run(self, state: ProcExState) -> ProcExState:
        videos_dir = self.cfg.dirs["videos"]
        os.makedirs(videos_dir, exist_ok=True)

        def _ancestry_key(scene_id: int):
            path = []
            while scene_id >= 100:
                path.append(scene_id % 100)
                scene_id = scene_id // 100
            path.append(scene_id)
            return tuple(reversed(path))

        scenes_with_clips = sorted(
            [
                s for s in state.scenes
                if s.clip_path
                and os.path.exists(s.clip_path)
                and os.path.getsize(s.clip_path) > 0
            ],
            key=lambda s: _ancestry_key(s.id)
        )

        if not scenes_with_clips:
            self._err(state, "No clips to assemble")
            return state

        self._log(f"Assembling {len(scenes_with_clips)} clips + audio...")

        res         = RESOLUTIONS[state.resolution]
        final_video = os.path.join(videos_dir, f"{state.topic_slug}.mp4")
        synced_dir  = os.path.join(videos_dir, "synced")
        os.makedirs(synced_dir, exist_ok=True)

        # Check whether per-scene audio files exist.
        # If a scene's audio file is missing but state.audio_path exists,
        # reconstruct the path using tts_audio_start offset so legacy mode
        # is never triggered just because a path string is stale/wrong.
        for s in scenes_with_clips:
            if s.tts_audio_path and not os.path.exists(s.tts_audio_path):
                # Path might be stale from a different working directory.
                # Try to resolve relative to current dir.
                candidate = os.path.abspath(s.tts_audio_path)
                if os.path.exists(candidate):
                    s.tts_audio_path = candidate

        has_per_scene_audio = any(
            s.tts_audio_path and os.path.exists(s.tts_audio_path)
            for s in scenes_with_clips
        )

        # Fallback: if no per-scene audio but we have the full combined audio,
        # reconstruct per-scene audio refs using tts_audio_start offsets so
        # we can still do segment-level sync rather than the crude legacy overlay.
        if not has_per_scene_audio and state.audio_path and os.path.exists(state.audio_path):
            self._log("Per-scene audio not found — reconstructing from combined audio + offsets")
            for s in scenes_with_clips:
                s.tts_audio_path = state.audio_path
                # tts_audio_start already carries the correct offset (set by TTSAgent)
            has_per_scene_audio = True

        if has_per_scene_audio:
            # ── Segment-level sync ────────────────────────────────────────
            synced_segments = self._sync_all_segments(
                scenes_with_clips, res, synced_dir
            )
            self._concat_clips(synced_segments, final_video)
            self._log(f"Segment-synced video -> {final_video}")
        else:
            # ── Legacy fallback ───────────────────────────────────────────
            self._log("WARNING: No per-scene audio paths — using legacy sync")
            silent     = os.path.join(videos_dir, f"{state.topic_slug}_silent.mp4")
            normalized = self._normalize_clips(scenes_with_clips, res, synced_dir)
            self._concat_clips(normalized, silent)
            if state.audio_path and os.path.exists(state.audio_path):
                self._overlay_audio(silent, state.audio_path, final_video)
            else:
                shutil.copy2(silent, final_video)

        # ── SRT subtitles ─────────────────────────────────────────────────
        srt_path = os.path.join(videos_dir, f"{state.topic_slug}.srt")
        try:
            self._generate_srt(state, srt_path)
            self._log(f"Subtitles -> {srt_path}")
        except Exception as e:
            self._log(f"SRT generation failed (non-critical): {e}")

        # ── Subtitle burn — skipped on Windows (libass not configured) ────
        if platform.system() != "Windows" and os.path.exists(srt_path):
            cc_video = os.path.join(videos_dir, f"{state.topic_slug}_cc.mp4")
            try:
                self._burn_subtitles(final_video, srt_path, cc_video)
                self._log(f"Captioned video -> {cc_video}")
            except Exception as e:
                self._log(f"Subtitle burn failed (non-critical): {e}")
        elif platform.system() == "Windows":
            self._log(
                "Subtitle burn skipped on Windows. "
                "Load subtitles in VLC: Subtitle -> Add Subtitle File"
            )

        state.final_video_path = final_video
        size_mb  = os.path.getsize(final_video) / 1_048_576
        duration = self._probe_duration(final_video)
        self._log(f"Done: {duration:.1f}s ({duration/60:.1f} min), {size_mb:.1f} MB")

        return state

    # ── Segment sync ──────────────────────────────────────────────────────────

    def _sync_all_segments(
        self, scenes: list[Scene], res, synced_dir: str
    ) -> list[str]:
        segments = []
        for scene in scenes:
            try:
                seg = self._sync_scene_segment(scene, res, synced_dir)
                segments.append(seg)
                clip_dur  = self._probe_duration(scene.clip_path)
                self._log(
                    f"Scene {scene.id}: synced "
                    f"(clip={clip_dur:.2f}s audio={scene.tts_duration:.2f}s)"
                )
            except Exception as e:
                self._log(f"Scene {scene.id}: sync failed ({e}), using raw clip")
                segments.append(scene.clip_path)
        return segments

    def _sync_scene_segment(
        self, scene: Scene, res, out_dir: str
    ) -> str:
        """
        Produce one muxed .mp4 where video and audio are the same duration.

        If the Manim clip is shorter than the TTS audio (common — the LLM
        rarely fills the exact duration), we freeze-extend the last frame.
        If longer, we trim.
        """
        clip_path  = scene.clip_path
        audio_path = scene.tts_audio_path
        audio_dur  = scene.tts_duration
        audio_start = getattr(scene, "tts_audio_start", 0.0) or 0.0
        clip_dur   = self._probe_duration(clip_path)
        seg_path   = os.path.join(out_dir, f"seg_{scene.id:02d}.mp4")

        if clip_dur  <= 0: raise ValueError(f"Cannot probe clip duration: {clip_path}")
        if audio_dur <= 0: raise ValueError(f"Audio duration is zero: {audio_path}")

        # Build audio input flags — seek into parent audio for subscene beats
        audio_input = ["-i", audio_path]
        audio_map_flags = ["-map", "1:a:0"]
        if audio_start > 0.01:
            # Subscene beat: slice the correct portion of the parent audio
            audio_input = ["-ss", f"{audio_start:.4f}", "-i", audio_path]

        target_w, target_h = res.width, res.height
        clip_info   = self._probe_clip(clip_path)
        needs_scale = (
            clip_info.get("width")  != target_w
            or clip_info.get("height") != target_h
            or clip_info.get("pix_fmt", "") not in ("yuv420p", "yuvj420p")
        )
        scale_filter = (
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:-1:-1:color=black,"
            if needs_scale else ""
        )

        gap = audio_dur - clip_dur

        if abs(gap) < 0.08:
            # Already matches — just normalize and mux
            vf = f"{scale_filter}fps=25,format=yuv420p" if scale_filter else "fps=25,format=yuv420p"
            cmd = [
                "ffmpeg", "-y",
                "-i", clip_path, *audio_input,
                "-vf", vf,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v:0", "-map", "1:a:0",
                "-t", f"{audio_dur:.4f}",
                seg_path,
            ]

        elif gap > 0:
            # Clip SHORT — freeze last frame for `gap` seconds
            self._log(
                f"Scene {scene.id}: clip short by {gap:.2f}s — "
                "freeze-extending last frame"
            )
            vf = (
                f"{scale_filter}"
                f"fps=25,format=yuv420p,"
                f"tpad=stop_mode=clone:stop_duration={gap:.4f}"
            )
            cmd = [
                "ffmpeg", "-y",
                "-i", clip_path, *audio_input,
                "-vf", vf,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v:0", "-map", "1:a:0",
                "-t", f"{audio_dur:.4f}",
                seg_path,
            ]

        else:
            # Clip LONG — trim to audio duration
            self._log(
                f"Scene {scene.id}: clip long by {-gap:.2f}s — "
                "trimming to audio duration"
            )
            vf = f"{scale_filter}fps=25,format=yuv420p" if scale_filter else "fps=25,format=yuv420p"
            cmd = [
                "ffmpeg", "-y",
                "-i", clip_path, *audio_input,
                "-vf", vf,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v:0", "-map", "1:a:0",
                "-t", f"{audio_dur:.4f}",
                seg_path,
            ]

        timeout = max(120, int(audio_dur * 20))  # raised from *10 — Windows overhead
        result  = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )

        if (
            result.returncode != 0
            or not os.path.exists(seg_path)
            or os.path.getsize(seg_path) == 0
        ):
            raise RuntimeError(f"Segment sync failed:\n{result.stderr[-800:]}")

        return seg_path

    # ── Normalize (legacy path) ────────────────────────────────────────────────

    def _normalize_clips(
        self, scenes: list[Scene], res, out_dir: str
    ) -> list[str]:
        target_w, target_h = res.width, res.height
        normalized = []
        for i, scene in enumerate(scenes):
            clip      = scene.clip_path
            norm_path = os.path.join(out_dir, f"norm_{i:02d}.mp4")
            info      = self._probe_clip(clip)

            if (
                info.get("width")  == target_w
                and info.get("height") == target_h
                and info.get("codec")  == "h264"
                and abs(info.get("fps", 0) - 25) < 1
                and info.get("pix_fmt", "") in ("yuv420p", "yuvj420p")
            ):
                shutil.copy2(clip, norm_path)
                normalized.append(norm_path)
                continue

            duration = info.get("duration", 30)
            cmd = [
                "ffmpeg", "-y", "-i", clip,
                "-vf", (
                    f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                    f"pad={target_w}:{target_h}:-1:-1:color=black,fps=25"
                ),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                norm_path,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=max(120, int(duration * 8)),
            )
            if result.returncode == 0 and os.path.getsize(norm_path) > 0:
                normalized.append(norm_path)
            else:
                self._log(f"Clip {i} normalization failed — using original")
                normalized.append(clip)
        return normalized

    # ── Concat ────────────────────────────────────────────────────────────────

    def _concat_clips(self, clips: list[str], output: str) -> None:
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

    # ── Legacy audio overlay ──────────────────────────────────────────────────

    def _overlay_audio(self, video: str, audio: str, output: str) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-i", video, "-i", audio,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0", "-shortest", output,
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
            h, m, sc, ms = (
                int(s // 3600), int((s % 3600) // 60),
                int(s % 60), int((s % 1) * 1000)
            )
            return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"

        WORDS_PER_SUB = 7
        chunks = []
        for i in range(0, len(state.all_timestamps), WORDS_PER_SUB):
            chunk = state.all_timestamps[i : i + WORDS_PER_SUB]
            if chunk:
                chunks.append((
                    chunk[0].start, chunk[-1].end,
                    " ".join(t.word for t in chunk),
                ))

        with open(srt_path, "w", encoding="utf-8") as f:
            for idx, (start, end, text) in enumerate(chunks, 1):
                f.write(f"{idx}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n\n")

    def _burn_subtitles(self, video: str, srt: str, output: str) -> None:
        srt_esc = srt.replace("\\", "/").replace(":", "\\:")
        cmd = [
            "ffmpeg", "-y", "-i", video,
            "-vf", (
                f"subtitles={srt_esc}:"
                "force_style='FontName=Arial,FontSize=22,"
                "PrimaryColour=&H00F0F0FF,OutlineColour=&H000A0A0F,"
                "Outline=2,Alignment=2,MarginV=40'"
            ),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy", output,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Subtitle burn failed: {result.stderr[-500:]}")

    # ── Probe ─────────────────────────────────────────────────────────────────

    def _probe_clip(self, clip: str) -> dict:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,codec_name,r_frame_rate,pix_fmt:format=duration",
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
                    if   k == "width":       info["width"]   = int(v)
                    elif k == "height":      info["height"]  = int(v)
                    elif k == "codec_name":  info["codec"]   = v
                    elif k == "pix_fmt":     info["pix_fmt"] = v
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

    @staticmethod
    def _probe_duration(path: str) -> float:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0