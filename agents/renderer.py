"""
agents/renderer.py
Renders each scene to a .mp4 clip.

Render-error feedback loop:
  When a MANIM scene fails to render, instead of immediately falling back to
  a black clip, the renderer calls ManimCoder.regenerate_for_render_error()
  with the exact Manim traceback. The coder rewrites the .py file and the
  renderer retries once. Only if that also fails does it use the emergency
  fallback (plain dark colour clip via FFmpeg).

This catches all Manim runtime errors that py_compile cannot: VGroup/Group
type mismatches, invalid ManimColor usage, undefined mobject methods, etc.
"""
from __future__ import annotations
import os
import subprocess
import shutil
from pathlib import Path
from state import ProcExState, Scene, VisualStrategy
from config import ProcExConfig, RESOLUTIONS, MANIM_PALETTE_BLOCK
from utils.llm_client import LLMClient
from utils.ken_burns import image_to_video_clip, cycle_effect
from agents.base_agent import BaseAgent
from agents.vlm_critic import VLMCritic, CriticResult


# Maximum number of regenerate-and-retry cycles per scene before black fallback
REGEN_RETRIES = 2


class RendererAgent(BaseAgent):
    name = "RendererAgent"

    def run(self, state: ProcExState) -> ProcExState:
        self._log(f"Rendering {len(state.scenes)} scene clips...")
        scenes_dir = self.cfg.dirs["scenes"]
        os.makedirs(scenes_dir, exist_ok=True)

        # Lazy-import ManimCoder to avoid circular imports at module load
        from agents.manim_coder import ManimCoder
        coder = ManimCoder(self.cfg, self.llm)

        for scene in state.scenes:
            clip_path = self._render_with_regen(
                scene, state.resolution, scenes_dir, coder, state.skill_pack
            )
            if clip_path:
                scene.clip_path = clip_path
                state.rendered_clips.append(clip_path)
                self._log(f"Scene {scene.id} -> {clip_path}")
            else:
                self._err(state, f"Scene {scene.id}: all render attempts failed")

        self._log(
            f"Rendered {len(state.rendered_clips)}/{len(state.scenes)} clips"
        )
        return state


    # ── Public single-attempt method (used by parallel queue runner) ──────────

    def render_scene_once(
        self,
        scene,
        resolution: str,
        out_dir: str,
    ) -> tuple:
        """
        Attempt to render one scene exactly once.
        Returns (clip_path, None) on success, (None, error_str) on failure.

        The parallel queue runner uses this instead of run() so it can route
        failures to the regen queue without any cross-agent callback inside
        the renderer itself.
        """
        try:
            clip_path = self._render_scene(scene, resolution, out_dir)
            return clip_path, None
        except Exception as e:
            return None, str(e)

    def render_emergency(
        self,
        scene,
        resolution: str,
        out_dir: str,
    ) -> str | None:
        """
        Produce a plain dark fallback clip when all regen attempts are exhausted.
        Returns clip path, or None if even the fallback fails.
        """
        try:
            return self._render_emergency_fallback(scene, resolution, out_dir)
        except Exception as e:
            self._log(f"Scene {scene.id}: emergency fallback failed: {e}")
            return None

    def render_with_critic(
        self,
        scene,
        resolution:  str,
        out_dir:     str,
        critic:      "VLMCritic | None" = None,
    ) -> tuple:
        """
        Render once, then run the VLM Critic if available.

        Returns (clip_path, error, critic_result) — caller unpacks all three.

        Flow:
          1. render_scene_once  → if fails, return (None, error, None)
          2. critic.inspect()   → CriticResult
             "ok"           → return (clip_path, None, result)
             "patched"      → re-render once with corrected code
             "reroute"      → return (clip_path, None, result) with result.reroute_frame
                               parallel_runner calls VisualDirector.reroute_scene()
                               and re-queues the scene for a fresh render
             "split_needed" → flag scene._split_recommended = True,
                               return original clip as fallback
        """
        clip_path, error = self.render_scene_once(scene, resolution, out_dir)

        if error or critic is None:
            return clip_path, error, None

        # ── VLM Critic inspection ─────────────────────────────────────────────
        from config import RESOLUTIONS
        res    = RESOLUTIONS.get(resolution, RESOLUTIONS["1080p"])
        aspect = res.aspect_ratio
        try:
            result: CriticResult = critic.inspect(clip_path, scene, aspect=aspect)
        except Exception as e:
            self._log(f"Scene {scene.id}: critic error (non-fatal): {e}")
            return clip_path, None, None

        if result.status == "ok":
            return clip_path, None, result

        if result.status == "reroute":
            self._log(
                f"Scene {scene.id}: Critic recommends reroute to VisualDirector — "
                f"{result.reason}"
            )
            return clip_path, None, result   # caller handles reroute loop

        if result.status == "split_needed":
            self._log(
                f"Scene {scene.id}: Critic recommends split — "
                f"{result.reason}. Flagging for orchestrator."
            )
            scene._split_recommended = True
            return clip_path, None, result

        if result.status == "patched":
            self._log(
                f"Scene {scene.id}: Critic patched {len(result.issues)} "
                f"collision(s) — re-rendering..."
            )
            # Purge stale Manim media output before re-rendering.
            # On Windows, Manim's caching can return the old clip even with
            # --disable_caching if the class name and output path are unchanged.
            # Deleting the previous output forces a clean render of the patched code.
            self._purge_manim_cache(scene, resolution)

            new_clip, new_err = self.render_scene_once(scene, resolution, out_dir)
            if new_err:
                self._log(
                    f"Scene {scene.id}: critic re-render failed ({new_err}) "
                    f"— keeping original"
                )
                return clip_path, None, result
            return new_clip, None, result

        return clip_path, None, result

    # ── Render with regeneration loop ─────────────────────────────────────────

    def _render_with_regen(
        self,
        scene: Scene,
        resolution: str,
        out_dir: str,
        coder,
        skill_pack: dict,
    ) -> str | None:
        """
        Attempt to render a scene.  On MANIM/TEXT_ANIMATION failure:
          1. Call ManimCoder.regenerate_for_render_error() with the traceback
          2. Retry rendering with the new .py file
          3. If still failing after REGEN_RETRIES, use the plain FFmpeg fallback
        IMAGE_GEN and HYBRID scenes skip the regen loop (no Manim involved).
        """
        is_manim = scene.visual_strategy in (
            VisualStrategy.MANIM, VisualStrategy.TEXT_ANIMATION
        )

        # Non-Manim scenes — one attempt, straight fallback on failure
        if not is_manim:
            try:
                return self._render_scene(scene, resolution, out_dir)
            except Exception as e:
                self._log(f"Scene {scene.id} (image) failed: {e}")
                return self._render_emergency_fallback(scene, resolution, out_dir)

        # Manim scenes — initial attempt + up to REGEN_RETRIES regenerations
        last_error = ""

        for attempt in range(1 + REGEN_RETRIES):
            try:
                return self._render_scene(scene, resolution, out_dir)

            except Exception as e:
                last_error = str(e)

                if attempt < REGEN_RETRIES:
                    self._log(
                        f"Scene {scene.id} render attempt {attempt+1} failed — "
                        f"regenerating code..."
                    )
                    try:
                        coder.regenerate_for_render_error(
                            scene, skill_pack, last_error
                        )
                        # Continue loop — next iteration retries with new file
                    except Exception as regen_err:
                        self._log(
                            f"Scene {scene.id} regeneration failed: {regen_err}"
                        )
                        break  # Skip remaining retries, go to fallback
                else:
                    self._log(
                        f"Scene {scene.id}: {REGEN_RETRIES} regeneration(s) exhausted"
                    )

        # All render attempts failed — use plain dark-background fallback
        self._log(f"Scene {scene.id}: using emergency fallback clip")
        try:
            return self._render_emergency_fallback(scene, resolution, out_dir)
        except Exception as fb_err:
            self._log(f"Scene {scene.id}: emergency fallback also failed: {fb_err}")
            return None

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def _render_scene(self, scene: Scene, resolution: str, out_dir: str) -> str:
        strategy = scene.visual_strategy
        if strategy in (VisualStrategy.MANIM, VisualStrategy.TEXT_ANIMATION):
            return self._render_manim(scene, resolution, out_dir)
        elif strategy == VisualStrategy.IMAGE_GEN:
            return self._render_ken_burns(scene, resolution, out_dir)
        elif strategy == VisualStrategy.IMAGE_MANIM_HYBRID:
            return self._render_ken_burns(scene, resolution, out_dir)  # retired — use ken burns
        else:
            raise ValueError(f"Unknown visual strategy: {strategy}")

    # ── Manim ─────────────────────────────────────────────────────────────────

    def _render_manim(self, scene: Scene, resolution: str, out_dir: str) -> str:
        if not scene.manim_file_path or not os.path.exists(scene.manim_file_path):
            raise FileNotFoundError(f"Manim file missing: {scene.manim_file_path}")

        res              = RESOLUTIONS[resolution]
        class_name       = scene.manim_class_name
        manim_media      = os.path.join(self.cfg.dirs["manim"], "media")
        scene_file_abs   = os.path.abspath(scene.manim_file_path)
        scene_dir        = os.path.dirname(scene_file_abs)
        manim_media_abs  = os.path.abspath(manim_media)

        # For portrait (9:16) resolutions, write a manim.cfg next to the
        # scene file. --pixel_width/--pixel_height are NOT CLI flags in
        # Manim Community — they are config-file-only options. frame_width
        # and frame_height have no CLI flag at all. The cfg file is the
        # only reliable way to set all four dimensions across all versions.
        cfg_path = os.path.join(scene_dir, "manim.cfg")
        if res.is_portrait:
            cfg_content = (
                "[CLI]\n"
                f"pixel_width = {res.width}\n"
                f"pixel_height = {res.height}\n"
                f"frame_width = {res.manim_frame_width}\n"
                f"frame_height = {res.manim_frame_height}\n"
            )
            with open(cfg_path, "w", encoding="utf-8") as cfg_f:
                cfg_f.write(cfg_content)
        else:
            # Remove any stale portrait cfg so landscape renders are clean
            if os.path.exists(cfg_path):
                os.remove(cfg_path)

        cmd = [
            "manim",
            res.manim_flag,
            "--media_dir", manim_media_abs,
            "--disable_caching",
            scene_file_abs,
            class_name,
        ]

        self._log(f"Scene {scene.id}: manim {res.manim_flag} {class_name}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.cfg.manim_timeout_secs,
            cwd=scene_dir,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Manim render failed:\n{result.stderr[-3000:]}")

        rendered = self._find_manim_output(manim_media_abs, class_name, res.manim_flag)
        if not rendered:
            raise FileNotFoundError(f"Manim output not found for {class_name}")

        dest = os.path.join(out_dir, f"scene_{scene.id:02d}.mp4")
        shutil.copy2(rendered, dest)
        return dest

    def _find_manim_output(
        self, media_dir: str, class_name: str, quality_flag: str
    ) -> str | None:
        """Search with pathlib.glob — handles Windows backslash paths correctly."""
        quality_map = {
            "-ql": "480p15", "-qm": "720p30",
            "-qh": "1080p60", "-qk": "2160p60",
        }
        quality    = quality_map.get(quality_flag, "1080p60")
        media_path = Path(media_dir)

        for pattern in [
            f"videos/**/{class_name}.mp4",
            f"videos/**/{quality}/{class_name}.mp4",
            f"**/{class_name}.mp4",
        ]:
            matches = list(media_path.glob(pattern))
            if matches:
                return str(sorted(matches, key=lambda p: p.stat().st_mtime)[-1])

        return None

    def _purge_manim_cache(self, scene: Scene, resolution: str) -> None:
        """
        Delete the stale Manim media output for this scene before a patch
        re-render. On Windows, Manim ignores --disable_caching when the class
        name and output path are unchanged — it returns the old clip silently.
        Deleting the previous output forces a genuine re-execution.
        """
        from config import RESOLUTIONS
        res          = RESOLUTIONS.get(resolution, RESOLUTIONS["1080p"])
        manim_media  = os.path.join(self.cfg.dirs["manim"], "media")
        manim_media_abs = os.path.abspath(manim_media)
        class_name   = scene.manim_class_name

        stale = self._find_manim_output(manim_media_abs, class_name, res.manim_flag)
        if stale and os.path.exists(stale):
            try:
                os.remove(stale)
                self._log(f"Scene {scene.id}: purged stale Manim cache → {stale}")
            except Exception as e:
                self._log(f"Scene {scene.id}: could not purge cache ({e}) — proceeding anyway")

    # ── Ken Burns ─────────────────────────────────────────────────────────────

    def _render_ken_burns(self, scene: Scene, resolution: str, out_dir: str) -> str:
        if not scene.image_paths:
            raise ValueError(f"Scene {scene.id}: no image paths for IMAGE_GEN")

        dest = os.path.join(out_dir, f"scene_{scene.id:02d}.mp4")
        image_to_video_clip(
            image_path  = scene.image_paths[0],
            duration    = scene.duration_seconds,
            output_path = dest,
            resolution  = resolution,
            effect      = cycle_effect(scene.id),
        )
        return dest

    # ── Hybrid ────────────────────────────────────────────────────────────────

    def _render_hybrid(self, scene: Scene, resolution: str, out_dir: str) -> str:
        res     = RESOLUTIONS[resolution]
        bg_clip = os.path.join(out_dir, f"scene_{scene.id:02d}_bg.mp4")

        if scene.image_paths:
            image_to_video_clip(
                image_path  = scene.image_paths[0],
                duration    = scene.duration_seconds,
                output_path = bg_clip,
                resolution  = resolution,
                effect      = "drift",
            )
        else:
            return self._render_manim(scene, resolution, out_dir)

        if scene.manim_file_path and os.path.exists(scene.manim_file_path):
            manim_clip = self._render_manim_transparent(scene, resolution, out_dir)
        else:
            dest = os.path.join(out_dir, f"scene_{scene.id:02d}.mp4")
            shutil.copy2(bg_clip, dest)
            return dest

        dest = os.path.join(out_dir, f"scene_{scene.id:02d}.mp4")
        cmd  = [
            "ffmpeg", "-y",
            "-i", bg_clip, "-i", manim_clip,
            "-filter_complex",
            f"[0:v][1:v]overlay=0:0:format=auto,scale={res.ffmpeg_scale}[v]",
            "-map", "[v]", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            dest,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=120
        )
        if result.returncode != 0:
            self._log(f"Hybrid composite failed — using bg only")
            shutil.copy2(bg_clip, dest)

        return dest

    def _render_manim_transparent(
        self, scene: Scene, resolution: str, out_dir: str
    ) -> str:
        res             = RESOLUTIONS[resolution]
        manim_media     = os.path.join(self.cfg.dirs["manim"], "media_transparent")
        dest            = os.path.join(out_dir, f"scene_{scene.id:02d}_overlay.mp4")
        scene_file_abs  = os.path.abspath(scene.manim_file_path)
        manim_media_abs = os.path.abspath(manim_media)

        cmd = [
            "manim", res.manim_flag, "--transparent",
            "--media_dir", manim_media_abs, "--disable_caching",
            scene_file_abs, scene.manim_class_name,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=self.cfg.manim_timeout_secs,
            cwd=os.path.dirname(scene_file_abs),
        )
        if result.returncode != 0:
            raise RuntimeError(f"Transparent Manim failed: {result.stderr[-1000:]}")

        rendered = self._find_manim_output(
            manim_media_abs, scene.manim_class_name, res.manim_flag
        )
        if rendered:
            shutil.copy2(rendered, dest)
            return dest

        raise FileNotFoundError("Transparent Manim output not found")

    # ── Emergency fallback ────────────────────────────────────────────────────

    def _render_emergency_fallback(
        self, scene: Scene, resolution: str, out_dir: str
    ) -> str:
        """
        Last-resort fallback when all Manim render attempts are exhausted.
        Uses ManimCoder's guaranteed-runnable title card (dark cinematic slide
        with scene narration excerpt) rather than a plain black screen.
        Falls back to an ffmpeg colour clip only if even that fails.
        """
        from agents.manim_coder import ManimCoder, _fallback_scene
        res  = RESOLUTIONS[resolution]
        dest = os.path.join(out_dir, f"scene_{scene.id:02d}.mp4")

        # Try Manim title card first
        try:
            class_name            = getattr(scene, "manim_class_name", f"Scene{scene.id:02d}")
            scene.manim_class_name = class_name
            fallback_code = _fallback_scene(class_name, scene)
            # Write to a dedicated fallback file so it doesn't clobber the broken code
            fb_path = os.path.join(
                self.cfg.dirs["manim"], f"scene_{scene.id:02d}_fallback.py"
            )
            coder = ManimCoder(self.cfg, None)
            coder._write_scene_file(fb_path, fallback_code, res=res)

            # Render the fallback file
            old_path = scene.manim_file_path
            scene.manim_file_path = fb_path
            clip = self._render_manim(scene, resolution, out_dir)
            scene.manim_file_path = old_path
            self._log(f"Scene {scene.id}: emergency fallback rendered via Manim title card")
            return clip
        except Exception as e:
            self._log(f"Scene {scene.id}: Manim fallback failed ({e}) — using ffmpeg colour clip")

        # True last resort: plain dark colour clip
        duration = max(5.0, scene.duration_seconds)
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=0x0A0A0F:size={res.width}x{res.height}:rate=25",
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            dest,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=60
        )
        if result.returncode != 0:
            raise RuntimeError(f"Emergency fallback failed: {result.stderr[-500:]}")
        return dest