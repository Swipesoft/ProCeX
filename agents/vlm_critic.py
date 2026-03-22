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
    get_zones,
    draw_grid_overlay,
    frame_to_base64,
    zone_manifest,
    zone_to_shift_hint,
    zone_to_manim_position,
)


# ── Thresholds ────────────────────────────────────────────────────────────────
CRITIC_DENSITY_THRESHOLD  = 2   # element count >= this triggers critic
CRITIC_MAX_PATCHES        = 5   # max correction actions per scene
PATCH_MAX_TOKENS          = 16384
VISION_MAX_TOKENS         = 16384
KEYFRAME_OFFSETS          = [0.75, 0.88, 0.97]  # peak density window
MAX_REROUTE_ATTEMPTS      = 2   # max times a scene can be sent back to VisualDirector
REROUTE_DENSITY_THRESHOLD = 7   # density score >= this → reroute rather than patch


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
    status:         str            # "ok" | "patched" | "reroute" | "split_needed" | "imagegen_fallback"
    issues:         list           = field(default_factory=list)
    patched_code:   Optional[str]  = None
    reason:         str            = ""
    # "reroute": peak-density frame passed back to VisualDirector for re-planning
    reroute_frame:  Optional[bytes] = None  # raw JPEG bytes of the worst frame


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
Topic:       {narration_hint}

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
5. ZONE NAMES (TITLE, MAIN, SIDEBAR, FOOTER, CENTER, etc.) are coordinate
   references ONLY. Never create Text("TITLE"), Text("MAIN"), Text("SIDEBAR")
   or any other zone name as a visible Manim object.
6. Return ONLY the complete corrected Python code — no explanations, no
   comments about what you changed, no preamble, no markdown fences.
   The very first character of your response must be the first character
   of the Python file (typically 'f', 'i', 'c', or '#').
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
            + ", ".join(f"[{v.severity}] {v.element} {v.from_zone}\u2192{v.to_zone}" for v in issues)
        )

        split_recommended = vision_result.get("split_recommended", False)
        density_int       = int(density) if str(density).isdigit() else 0
        reroute_attempts  = getattr(scene, "critic_reroute_attempts", 0) or 0

        # ── Reroute decision ──────────────────────────────────────────────────
        # The Critic decides the remediation path here. Policy lives only here —
        # VisualDirector.reroute_scene() receives the frame and re-plans freely.
        #
        # Reroute when: density is structurally too high for repositioning alone,
        # OR split is explicitly recommended, AND budget allows.
        # Patch when: collision is a positional fix (low-medium density).
        # Split (final) when: reroute budget exhausted.
        # ── Depth gate — N ≤ K ≤ 2N boundary ────────────────────────────────
        # Scenes at split_depth >= 1 are already children of a split.
        # Allowing them to split again would break the 2N scene count ceiling
        # and cause runaway API spend. Instead, flag for ImageGen fallback:
        # the image generator receives rich context and produces a static frame
        # that respects the aspect ratio and TikTok safe zones.
        scene_split_depth = getattr(scene, "split_depth", 0)

        if scene_split_depth >= 1 and (
            split_recommended or density_int >= REROUTE_DENSITY_THRESHOLD
        ):
            peak_frame = frames[-1] if frames else None
            self._log(
                f"Scene {scene.id}: split_depth={scene_split_depth} — "
                f"further splitting blocked (N≤K≤2N). Flagging for ImageGen fallback."
            )
            return CriticResult(
                status        = "imagegen_fallback",
                issues        = issues,
                reason        = reasoning or "depth limit reached — ImageGen handles this scene",
                reroute_frame = peak_frame,
            )

        should_reroute = (
            split_recommended or density_int >= REROUTE_DENSITY_THRESHOLD
        ) and reroute_attempts < MAX_REROUTE_ATTEMPTS

        if should_reroute:
            # Pass the peak-density frame (last = 97%) back to VisualDirector.
            # No issue list — VisualDirector gets full creative freedom to
            # re-plan layout or split into beats as it sees fit.
            peak_frame = frames[-1] if frames else None
            self._log(
                f"Scene {scene.id}: rerouting to VisualDirector "
                f"(density={density}/10, attempts={reroute_attempts+1}/{MAX_REROUTE_ATTEMPTS})"
            )
            return CriticResult(
                status        = "reroute",
                issues        = issues,
                reason        = reasoning or "structural density overflow",
                reroute_frame = peak_frame,
            )

        if split_recommended and reroute_attempts >= MAX_REROUTE_ATTEMPTS:
            # Budget exhausted — pass peak frame so orchestrator can force PATH B split
            peak_frame = frames[-1] if frames else None
            self._log(
                f"Scene {scene.id}: reroute budget exhausted ({reroute_attempts}/{MAX_REROUTE_ATTEMPTS}) "
                f"— flagging as split_needed"
            )
            return CriticResult(
                status        = "split_needed",
                issues        = issues,
                reason        = reasoning or "density overflow, reroute budget exhausted",
                reroute_frame = peak_frame,
            )

        # ── Guard: no point patching if no actionable issues ─────────────────
        if not issues:
            self._log(f"Scene {scene.id}: collision detected but no actionable issues — passing as-is")
            return CriticResult(
                status = "ok",
                reason = "collision detected but no issues to fix",
            )

        # ── Stage 2: Claude patches the code ─────────────────────────────────
        # Reached when density is moderate and collision is positional.
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

        self._log(f"Scene {scene.id}: {len(issues)} collision(s) patched → queued for re-render")
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
          - Scene is Manim-based (TEXT_ANIMATION, MANIM)
          - Element count exceeds threshold
          - Not a fallback title card (those have nothing to fix)
        """
        manim_types = {
            VisualStrategy.MANIM,
            VisualStrategy.TEXT_ANIMATION,
        }
        if scene.visual_strategy not in manim_types:
            self._log(f"Scene {scene.id}: skipped (strategy={scene.visual_strategy.value})")
            return False

        element_count = getattr(scene, "element_count", 0) or 0
        if element_count < CRITIC_DENSITY_THRESHOLD:
            self._log(f"Scene {scene.id}: skipped (elements={element_count}, threshold={CRITIC_DENSITY_THRESHOLD})")
            return False

        # Skip if this is a fallback title card — clip is too short to be real Manim
        if scene.manim_file_path and os.path.exists(scene.manim_file_path):
            try:
                with open(scene.manim_file_path, encoding="utf-8") as f:
                    code = f.read()
                if "fallback" in code.lower() or (code.count("self.play") <= 2 and "freeze" not in code):
                    pass  # real scene — continue
            except Exception:
                pass

        self._log(f"Scene {scene.id}: inspecting (strategy={scene.visual_strategy.value}, elements={element_count})")
        return True

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
        # Compact zone list — names only, saves tokens vs full manifest table
        zones      = get_zones(aspect)
        zone_names = ", ".join(zones.keys())

        user_prompt = VISION_USER_TMPL.format(
            title          = f"Scene {scene.id}",
            description    = scene.visual_reasoning or (scene.narration_text or "")[:120],
            narration_hint = (scene.narration_text or "")[:200],
            zone_manifest  = f"Available zones: {zone_names}",
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
            # Strip markdown fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
            raw = re.sub(r"```\s*$",           "", raw.strip(), flags=re.MULTILINE)
            raw = raw.strip()

            # Direct parse
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass

            # Use json-repair to handle all malformed JSON cases:
            # truncation, missing commas, single quotes, trailing commas, etc.
            from json_repair import repair_json
            import re as _re
            try:
                result = json.loads(repair_json(raw, ensure_ascii=False))
                if isinstance(result, dict):
                    self._log("Vision JSON repaired via json-repair")
                    return result
            except Exception:
                pass

            # Last resort: regex extraction of boolean fields from severely malformed output
            collision = bool(_re.search(r'"collision_detected"\s*:\s*true', raw, _re.I))
            score_m   = _re.search(r'"density_score"\s*:\s*(\d+)', raw)
            score     = int(score_m.group(1)) if score_m else 0
            self._log(f"Vision JSON fallback parse: collision={collision} score={score}")
            return {
                "collision_detected": collision and score >= 7,
                "split_recommended":  False,
                "density_score":      score,
                "issues":             [],
                "reasoning":          "Partial parse — truncated response",
            }

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
            # Strip markdown fences
            patched = re.sub(r"^```(?:python)?\s*", "", patched.strip(), flags=re.MULTILINE)
            patched = re.sub(r"```\s*$",             "", patched.strip(), flags=re.MULTILINE)
            patched = patched.strip()

            # Strip any leading reasoning/prose lines before the first Python line.
            # Claude sometimes thinks out loud before the code ("Let me interpret...").
            # Any line before the first from/import/class/# is prose — remove it.
            lines = patched.splitlines()
            code_start = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if (stripped.startswith("from ")
                        or stripped.startswith("import ")
                        or stripped.startswith("class ")
                        or stripped.startswith("#")
                        or stripped == ""):
                    code_start = i
                    break
            patched = "\n".join(lines[code_start:]).strip()

            # Final guard: verify the result is valid Python before returning.
            # If it fails, log and return None so the original code is kept.
            try:
                compile(patched, "<patch>", "exec")
            except SyntaxError as se:
                self._log(f"Code patch syntax error after strip ({se}) — discarding patch")
                return None

            return patched

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