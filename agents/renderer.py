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


# Maximum number of regenerate-and-retry cycles per scene before black fallback
REGEN_RETRIES = 5


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
            return self._render_hybrid(scene, resolution, out_dir)
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
        """Plain dark colour clip — no drawtext to avoid Windows font issues."""
        res      = RESOLUTIONS[resolution]
        dest     = os.path.join(out_dir, f"scene_{scene.id:02d}.mp4")
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