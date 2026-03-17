"""
agents/vlm_critic.py
VLM-powered spatial layout Critic for rendered Manim scenes.

Inspired by the Critic in Code2Video (Chen et al., 2025) but adapted for
ProcEx's architecture and extended with a second remediation path: temporal
scene splitting for density-overflow collisions that repositioning alone
cannot solve.

Pipeline position:
  RendererAgent.render_scene_once() → SUCCESS
        ↓
  VLMCritic.inspect(clip_path, scene, manim_code)
        ↓ CriticResult
  .status == "ok"           → pass through, no action
  .status == "patched"      → Claude patches the code → re-render once
  .status == "split_needed" → scene flagged; parallel_runner handles split

Two-LLM handoff:
  1. Gemini 2.5 (vision)  — looks at the annotated frame, identifies
                            spatial collisions and their zones
  2. Claude Sonnet (code) — receives Gemini's diagnosis + original code
                            and produces the coordinate-corrected code patch

The split between "see" and "fix" is deliberate: Gemini is stronger at
visual grounding; Claude is stronger at precise Manim code manipulation.

Trigger gate:
  The Critic runs only when scene.element_count (set by VisualDirector)
  exceeds CRITIC_DENSITY_THRESHOLD, OR when the scene type is HYBRID.
  Clean single-element scenes are skipped to avoid unnecessary API cost.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agents.base_agent import BaseAgent
from config import ProcExConfig
from state import ProcExState, Scene, VisualStrategy
from utils.llm_client import LLMClient
from utils.spatial_grid import (
    ZONES,
    draw_grid_overlay,
    frame_to_base64,
    zone_manifest,
    zone_to_shift_hint,
    zone_to_manim_position,
)


# ── Thresholds ────────────────────────────────────────────────────────────────
CRITIC_DENSITY_THRESHOLD = 2   # element count >= this triggers critic (was 3)
CRITIC_MAX_PATCHES       = 5   # max correction actions per scene
PATCH_MAX_TOKENS         = 4096
VISION_MAX_TOKENS        = 1024
KEYFRAME_OFFSETS         = [0.25, 0.50, 0.75]  # sample at 25%, 50%, 75% of clip


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class CriticIssue:
    element:       str   # description of the colliding element
    from_zone:     str   # zone where it currently is (or "UNKNOWN")
    to_zone:       str   # zone where it should go
    action:        str   # "move" | "resize" | "remove_overlap"
    severity:      str   # "critical" | "moderate"


@dataclass
class CriticResult:
    status:         str            # "ok" | "patched" | "split_needed"
    issues:         list           = field(default_factory=list)
    patched_code:   Optional[str]  = None
    reason:         str            = ""   # explanation for logging


# ── Prompts ───────────────────────────────────────────────────────────────────

VISION_SYSTEM = """\
You are an expert visual layout critic for educational animation videos.
Your job is to inspect a rendered Manim animation frame and identify spatial
collisions — elements that overlap, occlude, or crowd each other to the point
where the content becomes hard to read.

The frame has a 6×6 grid overlay with labelled zone names. Use these zones
to precisely describe where each colliding element currently sits and where
it should move to resolve the collision.

Your output must be strictly valid JSON — no preamble, no markdown fences.
"""

VISION_USER_TMPL = """\
SCENE CONTEXT
=============
Title:       {title}
Description: {description}
Topic:       {narration_hint}

ZONE REFERENCE TABLE
====================
{zone_manifest}

TASK
====
Examine the annotated frame carefully.

1. Identify any elements that:
   - Overlap or occlude other elements
   - Are positioned outside their natural reading area
   - Are so densely packed that text is unreadable
   - Float over titles or footers unintentionally

2. Decide: can this be fixed by repositioning elements within the frame?
   If yes → list corrections.
   If the frame is simply too content-dense for one scene (>5 teaching points
   crammed into one view with no spatial resolution possible) → recommend split.

Respond ONLY with this JSON schema:
{{
  "collision_detected": true | false,
  "split_recommended":  true | false,
  "density_score":      0-10,
  "issues": [
    {{
      "element":    "<brief description of the colliding element>",
      "from_zone":  "<zone name where it currently sits>",
      "to_zone":    "<zone name where it should go>",
      "action":     "move" | "resize" | "remove_overlap",
      "severity":   "critical" | "moderate"
    }}
  ],
  "reasoning": "<one sentence summary>"
}}

If collision_detected is false, issues must be an empty list.
If split_recommended is true, still list whatever corrections would help even
after splitting (the two sub-scenes will each benefit from the guidance).
"""

PATCHER_SYSTEM = """\
You are an expert Manim Python developer. You will receive a Manim scene
class and a list of spatial correction instructions. Your job is to apply
the corrections by modifying only the positional values of the affected
elements — do not change animation logic, timing, colour, or content.

Rules:
1. Use .move_to(np.array([x, y, 0])) to reposition elements.
   Import numpy as np at the top if not already present.
2. Apply each correction as a targeted change — do not refactor unrelated code.
3. Ensure no two elements share the same Manim coordinate center.
4. If an element is in a VGroup, adjust the VGroup position, not the child.
5. Return ONLY the complete corrected Python code — no explanations, no fences.
"""

PATCHER_USER_TMPL = """\
ORIGINAL MANIM CODE
===================
{original_code}

SPATIAL CORRECTIONS TO APPLY
=============================
{correction_instructions}

ZONE COORDINATE LOOKUP
======================
{zone_manifest}

Apply all corrections and return the complete corrected Python code.
"""


class VLMCritic(BaseAgent):
    """
    Inspect a rendered scene clip for spatial collisions and return a
    CriticResult with either a patched Manim code string or a split flag.
    Called via inspect(), not run() — run() is a guard.
    """
    name = "VLMCritic"

    def run(self, state: ProcExState) -> ProcExState:
        raise NotImplementedError(
            "VLMCritic is called via inspect(clip_path, scene, manim_code), not run(state)."
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def inspect(
        self,
        clip_path:    str,
        scene:        Scene,
        manim_code:   Optional[str] = None,
        aspect:       str           = "16:9",
    ) -> CriticResult:
        """
        Main entry point. Called by RendererAgent after a successful render.

        clip_path:  path to the rendered .mp4 clip
        scene:      the Scene object (for context: title, description, type)
        manim_code: raw Manim Python source; if None, read from scene.manim_file_path

        Returns CriticResult.
        """
        # ── Gate: skip cheap/simple scenes ───────────────────────────────────
        if not self._should_inspect(scene):
            self._log(
                f"Scene {scene.id}: skipped "
                f"(strategy={scene.visual_strategy.value}, "
                f"elements={getattr(scene, 'element_count', 0)}, "
                f"threshold={CRITIC_DENSITY_THRESHOLD})"
            )
            return CriticResult(status="ok", reason="below density threshold")

        self._log(
            f"Scene {scene.id}: inspecting "
            f"(strategy={scene.visual_strategy.value}, "
            f"elements={getattr(scene, 'element_count', 0)})"
        )

        # ── Load code ─────────────────────────────────────────────────────────
        code = manim_code or self._load_code(scene)
        if not code:
            self._log(f"Scene {scene.id}: no Manim code found — skipping")
            return CriticResult(status="ok", reason="no Manim code to patch")

        # ── Extract keyframe ──────────────────────────────────────────────────
        self._log(f"Scene {scene.id}: extracting keyframes at {[int(o*100) for o in KEYFRAME_OFFSETS]}% of clip")
        frames = self._extract_keyframes(clip_path)
        if not frames:
            self._log(f"Scene {scene.id}: keyframe extraction failed — skipping critic")
            return CriticResult(status="ok", reason="keyframe extraction failed")

        # ── Annotate frames with grid overlay ────────────────────────────────
        annotated_frames = []
        for fb in frames:
            try:
                annotated_frames.append(draw_grid_overlay(fb, aspect=aspect))
            except Exception as e:
                self._log(f"Scene {scene.id}: grid overlay failed ({e}) — using raw frame")
                annotated_frames.append(fb)
        self._log(f"Scene {scene.id}: grid overlay applied to {len(annotated_frames)} frame(s)")

        # ── Stage 1: Gemini inspects all frames, worst result wins ───────────
        self._log(f"Scene {scene.id}: [stage 1] sending {len(annotated_frames)} frame(s) to Gemini vision...")
        vision_result = self._run_vision_analysis_multi(annotated_frames, scene, aspect=aspect)
        if vision_result is None:
            self._log(f"Scene {scene.id}: vision analysis failed gracefully — passing")
            return CriticResult(status="ok", reason="vision analysis failed gracefully")

        density = vision_result.get("density_score", "?")
        reasoning = vision_result.get("reasoning", "")
        collision = vision_result.get("collision_detected", False)
        self._log(
            f"Scene {scene.id}: Gemini result — "
            f"collision={collision}, density={density}/10, reason='{reasoning}'"
        )

        if not collision:
            self._log(f"Scene {scene.id}: no collisions detected — passing")
            return CriticResult(
                status  = "ok",
                reason  = reasoning or "no collisions detected",
            )

        issues = [
            CriticIssue(
                element   = i.get("element", "unknown element"),
                from_zone = i.get("from_zone", "UNKNOWN"),
                to_zone   = i.get("to_zone", "MAIN"),
                action    = i.get("action", "move"),
                severity  = i.get("severity", "moderate"),
            )
            for i in vision_result.get("issues", [])[:CRITIC_MAX_PATCHES]
        ]
        self._log(
            f"Scene {scene.id}: {len(issues)} issue(s) identified — "
            + ", ".join(f"[{v.severity}] {v.element} {v.from_zone}→{v.to_zone}" for v in issues)
        )

        # ── Split recommendation ──────────────────────────────────────────────
        if vision_result.get("split_recommended", False):
            self._log(
                f"Scene {scene.id}: split recommended — "
                f"density={density}/10, reason='{reasoning}'"
            )
            return CriticResult(
                status  = "split_needed",
                issues  = issues,
                reason  = reasoning or "density overflow",
            )

        # ── Stage 2: Claude patches the code ─────────────────────────────────
        self._log(f"Scene {scene.id}: [stage 2] sending {len(issues)} correction(s) to Claude patcher...")
        patched = self._run_code_patch(code, issues, scene, aspect=aspect)
        if not patched or patched == code:
            self._log(f"Scene {scene.id}: patch produced no change — passing as-is")
            return CriticResult(
                status  = "ok",
                issues  = issues,
                reason  = "patch produced no change",
            )

        # Write patched code back to the scene file
        if scene.manim_file_path and os.path.exists(scene.manim_file_path):
            try:
                with open(scene.manim_file_path, "w", encoding="utf-8") as f:
                    f.write(patched)
                self._log(f"Scene {scene.id}: patched code written to {scene.manim_file_path}")
            except Exception as e:
                self._log(f"Scene {scene.id}: could not write patched code — {e}")
                return CriticResult(status="ok", reason=f"file write failed: {e}")

        self._log(
            f"Scene {scene.id}: {len(issues)} collision(s) patched → queued for re-render"
        )
        return CriticResult(
            status       = "patched",
            issues       = issues,
            patched_code = patched,
            reason       = reasoning or "collisions corrected",
        )

    # ── Gate ──────────────────────────────────────────────────────────────────

    def _should_inspect(self, scene: Scene) -> bool:
        """
        Run the Critic only when:
          - Scene is Manim-based (TEXT_ANIMATION, MANIM, or HYBRID)
          - Element count exceeds threshold, OR scene is HYBRID
            (overlays always risk collision)
        """
        manim_types = {
            VisualStrategy.MANIM,
            VisualStrategy.TEXT_ANIMATION,
        }
        if scene.visual_strategy not in manim_types:
            return False

        element_count = getattr(scene, "element_count", 0) or 0
        return element_count >= CRITIC_DENSITY_THRESHOLD

    # ── Keyframe extraction ───────────────────────────────────────────────────

    def _extract_keyframes(self, clip_path: str) -> list[bytes]:
        """
        Extract frames at each KEYFRAME_OFFSETS percentage of clip duration.
        Returns list of JPEG byte strings; empty list on total failure.
        """
        if not os.path.exists(clip_path):
            return []

        # Get duration once
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", clip_path],
                capture_output=True, text=True, timeout=15,
            )
            duration = float(probe.stdout.strip() or "5")
        except Exception:
            duration = 5.0

        frames = []
        for offset in KEYFRAME_OFFSETS:
            seek_time = max(0.1, duration * offset)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                result = subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(seek_time), "-i", clip_path,
                     "-vframes", "1", "-q:v", "3", tmp_path],
                    capture_output=True, text=True, timeout=20,
                )
                if result.returncode == 0 and os.path.exists(tmp_path):
                    with open(tmp_path, "rb") as f:
                        frames.append(f.read())
            except Exception as e:
                self._log(f"Keyframe extraction error at {offset*100:.0f}%: {e}")
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        return frames

    # ── Vision analysis (Gemini) ──────────────────────────────────────────────

    def _run_vision_analysis_multi(
        self, frames: list[bytes], scene: Scene, aspect: str = "16:9"
    ) -> Optional[dict]:
        """
        Run vision analysis on each frame; return the worst result
        (highest density_score / any collision detected).
        'Worst wins' — a collision at any point in the animation counts.
        """
        results = []
        for i, frame in enumerate(frames):
            r = self._run_vision_analysis(frame, scene, aspect=aspect)
            if r is not None:
                results.append((i, r))

        if not results:
            return None

        # If any frame has a collision, return the one with highest density score
        collisions = [(i, r) for i, r in results if r.get("collision_detected", False)]
        if collisions:
            worst = max(collisions, key=lambda x: x[1].get("density_score", 0))
            self._log(
                f"Scene {scene.id}: collision found in frame {worst[0]+1}/{len(frames)} "
                f"(density={worst[1].get('density_score','?')}/10)"
            )
            return worst[1]

        # All frames clean — return the midpoint result
        mid = results[len(results) // 2]
        return mid[1]

    def _run_vision_analysis(self, frame_bytes: bytes, scene: Scene, aspect: str = "16:9") -> Optional[dict]:
        """
        Send annotated frame to Gemini vision. Returns parsed dict or None.
        """
        user_prompt = VISION_USER_TMPL.format(
            title          = scene.title or "Untitled Scene",
            description    = scene.description or "No description available",
            narration_hint = (scene.narration_text or "")[:300],
            zone_manifest  = zone_manifest(aspect),
        )

        try:
            raw = self.llm.complete_vision(
                system_prompt    = VISION_SYSTEM,
                user_prompt      = user_prompt,
                image_bytes      = frame_bytes,
                image_mime       = "image/jpeg",
                max_tokens       = VISION_MAX_TOKENS,
                temperature      = 0.1,
                primary_provider = "gemini",
            )
            # Strip any accidental markdown fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
            raw = re.sub(r"```\s*$",           "", raw.strip(), flags=re.MULTILINE)
            return json.loads(raw.strip())

        except json.JSONDecodeError as e:
            self._log(f"Vision JSON parse error: {e}")
            return None
        except Exception as e:
            self._log(f"Vision analysis error: {e}")
            return None

    # ── Code patching (Claude) ────────────────────────────────────────────────

    def _run_code_patch(
        self,
        original_code: str,
        issues:        list[CriticIssue],
        scene:         Scene,
        aspect:        str = "16:9",
    ) -> Optional[str]:
        """
        Hand correction instructions to Claude (code-strong) to produce a
        patched Manim file.
        """
        correction_lines = []
        for i, issue in enumerate(issues, 1):
            hint = zone_to_shift_hint(issue.from_zone, issue.to_zone, aspect=aspect)
            pos  = zone_to_manim_position(issue.to_zone, aspect=aspect)
            correction_lines.append(
                f"{i}. [{issue.severity.upper()}] Element: '{issue.element}'\n"
                f"   Action: {issue.action}\n"
                f"   Instruction: {hint}\n"
                f"   Manim position call: <element>{pos}"
            )

        correction_block = "\n\n".join(correction_lines)

        user_prompt = PATCHER_USER_TMPL.format(
            original_code          = original_code,
            correction_instructions= correction_block,
            zone_manifest          = zone_manifest(aspect),
        )

        try:
            patched = self.llm.complete(
                PATCHER_SYSTEM,
                user_prompt,
                max_tokens       = PATCH_MAX_TOKENS,
                temperature      = 0.2,
                primary_provider = "claude",
            )
            # Strip markdown fences if model wraps in ```python
            patched = re.sub(r"^```(?:python)?\s*", "", patched.strip(), flags=re.MULTILINE)
            patched = re.sub(r"```\s*$",             "", patched.strip(), flags=re.MULTILINE)
            return patched.strip()

        except Exception as e:
            self._log(f"Code patch error: {e}")
            return None

    # ── Load code from disk ───────────────────────────────────────────────────

    def _load_code(self, scene: Scene) -> Optional[str]:
        path = getattr(scene, "manim_file_path", None)
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
        return None