"""
agents/manim_coder.py
Generates runnable Manim Community Python code for MANIM / TEXT_ANIMATION scenes.

Sync approach: anchor-driven timing
  Instead of "make animations sum to duration_seconds" (which LLMs do poorly),
  we extract 5-7 animation anchors from word timestamps — each is a scene-relative
  time t at which the narrator speaks a key word. The prompt gives the LLM a
  pre-computed self.wait() skeleton so it only has to fill in the visual objects.
  Remaining drift is handled by the AssemblerAgent's freeze-extend safety net.

Validation pipeline per scene (max_llm_retries attempts before fallback):
  1. LLM generates class code
  2. _clean_code()         — strip fences, remove duplicate imports
  3. _validate_structure() — class name + construct method present
  4. _syntax_check()       — py_compile on full assembled file
  5. Fallback              — guaranteed-runnable cinematic title card

Render-error feedback: RendererAgent calls regenerate_for_render_error() when
Manim crashes at runtime. Exact traceback injected into next prompt.
"""
from __future__ import annotations
import os
import re
import py_compile
import tempfile
from state import ProcExState, Scene, VisualStrategy
from config import ProcExConfig, MANIM_PALETTE_BLOCK
from utils.llm_client import LLMClient
from utils.timestamp_utils import (
    extract_animation_anchors,
    anchors_to_prompt_block,
)
from agents.base_agent import BaseAgent


# ── File header injected at top of every generated scene ──────────────────────
MANIM_HEADER = '''from manim import *
from manim.utils.color import ManimColor
import numpy as np

{palette}

config.background_color = BG
config.frame_rate = 25
'''

# ── System prompt ──────────────────────────────────────────────────────────────
CODER_SYSTEM = """You are an expert Manim Community (v0.18+) code generator.
You write cinematic animations for educational videos, tightly synced to narration.

CORE RULES
==========
1. Output ONLY the Python class body — no imports, no config lines (already in file).
2. Class name: exactly as specified in scene_class_name.
3. class SceneN(Scene): with def construct(self): as the only method.
4. First line of construct: self.camera.background_color = BG
5. Palette constants: BG, WHITE, CYAN, PURPLE, ORANGE, GREEN, GOLD, DIM
   These are ManimColor objects — pass them directly to color=, fill_color=, etc.
6. DO NOT use: ImageMobject, SVGMobject, ThreeDScene, Surface, external files.
7. Equations: always MathTex(r"...") not Tex, always raw strings.
8. No TODO, no "...", no placeholder comments, no unfinished methods.
9. Always end with self.wait(0.5).

SYNC RULE — THIS IS THE MOST IMPORTANT RULE
============================================
You are given ANIMATION ANCHORS — a table of (t, word, action) and a
pre-computed TIMING SKELETON showing exact self.wait() values.

FOLLOW THE SKELETON EXACTLY:
  - Copy every self.wait(N.NN) line verbatim
  - Replace each self.play(...) comment with real Manim code
  - Do NOT add extra self.wait() calls between anchors (this breaks sync)
  - Do NOT change any wait duration
  - The waits are pre-calculated from real TTS audio timestamps

The result: each animation fires at the exact second the narrator speaks
the corresponding key word. This is how audio-visual sync is achieved.

VGROUP vs GROUP — CRITICAL
==========================
VGroup only accepts VMobject subclasses (Text, MathTex, Square, Circle,
Line, Arrow, Dot, etc.). Use Group() when mixing unknown types.
CORRECT: VGroup(Text("a"), MathTex(r"x^2"), Square())
WRONG:   VGroup(some_generic_mobject)  ← crashes with TypeError

LAYOUT RULES
============
L1. Frame: 14.22 wide x 8 tall. Keep x in [-5.5, 5.5], y in [-3.3, 3.3].
L2. Clear screen between major sections:
      self.play(FadeOut(Group(*self.mobjects)), run_time=0.5)
L3. Text sizes: titles ≤44, body=32, captions=24, never >52.
    Over 50 chars: Text("...", font_size=28, width=11)
L4. MathTex with >2 terms: always .scale(0.85)
L5. Multi-element layout: use Group/VGroup + .arrange(), not manual shift().
L6. Never draw new content on top of existing — FadeOut first.
L7. interpolate_color(BG, CYAN, 0.5) ← CORRECT (both are ManimColor objects)
    interpolate_color("#0A0A0F", CYAN, 0.5) ← WRONG (string crashes)

ANIMATION STYLE
===============
- Entrances: FadeIn(obj, shift=UP*0.3) or Write()
- Exits: always explicit FadeOut() — never leave objects stranded
- Camera moves require MovingCameraScene (change Scene → MovingCameraScene)
- Transform(old, new) morphs in place

OUTPUT: Return ONLY the class definition. No fences, no explanation.
"""


# ── Fallback: guaranteed-runnable cinematic title card ────────────────────────
def _fallback_scene(class_name: str, scene: Scene) -> str:
    title_raw = scene.narration_text[:55].replace("\\", "").replace('"', "'")
    wait_dur  = max(1.0, scene.duration_seconds - 3.5)
    return f'''class {class_name}(Scene):
    def construct(self):
        self.camera.background_color = BG

        label = Text("Scene {scene.id}", color=DIM, font_size=22)
        label.to_corner(UL, buff=0.4)

        title = Text("{title_raw}...", color=WHITE, font_size=34, width=11)
        title.move_to(ORIGIN)

        accent = Line(LEFT * 4, RIGHT * 4, color=CYAN, stroke_width=2)
        accent.next_to(title, DOWN, buff=0.35)

        self.play(FadeIn(label, shift=DOWN*0.2), Write(title), run_time=1.8)
        self.play(GrowFromCenter(accent), run_time=0.7)
        self.wait({wait_dur:.1f})
        self.play(FadeOut(Group(label, title, accent)), run_time=0.8)
        self.wait(0.5)
'''


def _build_coder_prompt(
    scene: Scene,
    skill: dict,
    error_context: str = "",
) -> str:
    """
    Build the ManimCoder prompt using anchor-driven timing.

    The prompt contains:
      - Scene metadata
      - Narration text (what is being said)
      - Visual brief (what to animate)
      - Animation anchors table (when to reveal each element)
      - Pre-computed timing skeleton (copy self.wait() values verbatim)
      - Domain elements available
    """
    manim_elements  = skill.get("manim_elements", [])
    domain_decoder  = skill.get("notation_decoder", "")

    # Extract anchors from this scene's word timestamps
    anchors = extract_animation_anchors(
        timestamps    = scene.timestamps,
        visual_prompt = scene.visual_prompt,
        n_anchors     = 6,
    )

    anchor_block = anchors_to_prompt_block(
        anchors  = anchors,
        duration = scene.duration_seconds,
    )

    error_block = ""
    if error_context:
        trimmed = error_context[-1200:] if len(error_context) > 1200 else error_context
        error_block = f"""
PREVIOUS ATTEMPT FAILED — READ AND FIX THIS ERROR:
---------------------------------------------------
{trimmed}
---------------------------------------------------
Fix the error above. Do NOT change the self.wait() values — those are correct.
"""

    domain_decoder_section = ""
    if domain_decoder:
        domain_decoder_section = f"""
═══════════════════════════════════════════════════════════════════════
DOMAIN NOTATION DECODER  (supplements the universal math decoder above)
═══════════════════════════════════════════════════════════════════════
The same principle applies here: narration uses plain spoken English.
Your job is to decode it into the canonical visual form for this domain.

{domain_decoder}
═══════════════════════════════════════════════════════════════════════
"""

    return f"""Generate Manim code for this scene.
{error_block}
scene_class_name: {scene.manim_class_name}
scene_id:         {scene.id}
duration_seconds: {scene.duration_seconds:.2f}

NARRATION (what the narrator says — this is the audio track):
{scene.narration_text}

VISUAL BRIEF (what to animate):
{scene.visual_prompt}

{anchor_block}
{domain_decoder_section}
DOMAIN ELEMENTS AVAILABLE:
{chr(10).join(f"  - {e}" for e in manim_elements)}

REMINDER: Copy the self.wait() values from the skeleton EXACTLY.
Replace each self.play(...) comment with real Manim animation code.
"""


class ManimCoder(BaseAgent):
    name = "ManimCoder"

    def run(self, state: ProcExState) -> ProcExState:
        manim_dir = self.cfg.dirs["manim"]
        os.makedirs(manim_dir, exist_ok=True)

        targets = [
            s for s in state.scenes
            if s.visual_strategy in (VisualStrategy.MANIM, VisualStrategy.TEXT_ANIMATION)
        ]

        self._log(f"Generating Manim code for {len(targets)} scenes...")

        for scene in targets:
            class_name             = f"Scene{scene.id:02d}"
            scene.manim_class_name = class_name
            scene_file             = os.path.join(manim_dir, f"scene_{scene.id:02d}.py")
            scene.manim_file_path  = scene_file

            code = self._generate_scene_code(scene, state.skill_pack)
            self._write_scene_file(scene_file, code)
            self._log(f"Scene {scene.id} -> {scene_file}")

        return state

    # ── Public: called by RendererAgent on render failure ─────────────────────

    def regenerate_for_render_error(
        self,
        scene: Scene,
        skill_pack: dict,
        render_error: str,
    ) -> bool:
        """
        Regenerate and rewrite the .py file for a scene that crashed at render.
        Injects the Manim runtime traceback so the LLM self-corrects.
        Returns True if regenerated successfully, False if fell back to title card.
        """
        self._log(f"Scene {scene.id}: render failed — regenerating with error context...")
        error_summary = self._summarise_render_error(render_error)
        code          = self._generate_scene_code(
            scene, skill_pack, initial_error=error_summary
        )
        is_fallback = f"Scene {scene.id}" in code[:80]
        self._write_scene_file(scene.manim_file_path, code)

        if is_fallback:
            self._log(f"Scene {scene.id}: regeneration fell back to title card")
            return False
        self._log(f"Scene {scene.id}: regenerated OK")
        return True

    # ── Code generation ────────────────────────────────────────────────────────

    def _generate_scene_code(
        self,
        scene: Scene,
        skill: dict,
        initial_error: str = "",
    ) -> str:
        header     = MANIM_HEADER.format(palette=MANIM_PALETTE_BLOCK)
        last_error = initial_error

        for attempt in range(self.cfg.max_llm_retries):
            try:
                raw  = self.llm.complete(
                    CODER_SYSTEM,
                    _build_coder_prompt(scene, skill, error_context=last_error),
                    json_mode=False,
                    max_tokens=6144,
                    temperature=0.3,
                    primary_provider="claude",
                )
                code = self._clean_code(raw, scene.manim_class_name)

                struct_ok, struct_err = self._validate_structure(code, scene.manim_class_name)
                if not struct_ok:
                    last_error = f"Structural validation failed: {struct_err}"
                    self._log(f"Scene {scene.id} attempt {attempt+1}: {last_error}")
                    continue

                syntax_ok, syntax_err = self._syntax_check(header, code)
                if not syntax_ok:
                    last_error = f"Python syntax error:\n{syntax_err}"
                    self._log(f"Scene {scene.id} attempt {attempt+1}: syntax error: {syntax_err[:80]}")
                    continue

                self._log(f"Scene {scene.id} attempt {attempt+1}: PASS")
                return code

            except Exception as e:
                last_error = str(e)
                self._log(f"Scene {scene.id} attempt {attempt+1} exception: {e}")

        self._log(f"Scene {scene.id}: all attempts failed — fallback title card")
        return _fallback_scene(scene.manim_class_name, scene)

    # ── File writer ────────────────────────────────────────────────────────────

    def _write_scene_file(self, scene_file: str, code: str) -> None:
        full = MANIM_HEADER.format(palette=MANIM_PALETTE_BLOCK) + "\n\n" + code
        with open(scene_file, "w", encoding="utf-8") as f:
            f.write(full)

    # ── Validators ────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_code(raw: str, class_name: str) -> str:
        raw = re.sub(r"^```(?:python)?\s*", "", raw.strip(), flags=re.MULTILINE)
        raw = re.sub(r"```\s*$",             "", raw.strip(), flags=re.MULTILINE)
        skip = {
            "from manim import", "import manim", "import numpy",
            "from manim.utils", "config.background", "config.frame_rate",
        }
        cleaned = [
            l for l in raw.splitlines()
            if not any(l.strip().startswith(s) for s in skip)
        ]
        return "\n".join(cleaned).strip()

    @staticmethod
    def _validate_structure(code: str, class_name: str) -> tuple[bool, str]:
        if f"class {class_name}" not in code:
            return False, f"class {class_name} not found"
        if "def construct(self)" not in code:
            return False, "def construct(self) not found"
        if len(code) < 100:
            return False, f"code too short ({len(code)} chars)"
        if code.count("...") > 3:
            return False, "too many '...' placeholders — incomplete"
        return True, ""

    @staticmethod
    def _syntax_check(header: str, code: str) -> tuple[bool, str]:
        full = header + "\n\n" + code
        tmp  = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        )
        try:
            tmp.write(full)
            tmp.close()
            py_compile.compile(tmp.name, doraise=True)
            return True, ""
        except py_compile.PyCompileError as e:
            msg = re.sub(r'File "[^"]+",? ?', "", str(e)).strip()
            return False, msg
        except Exception as e:
            return False, str(e)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    @staticmethod
    def _summarise_render_error(error: str) -> str:
        lines      = error.splitlines()
        user_lines = [l for l in lines if "scene_" in l and ".py" in l]
        error_type = ""
        for line in reversed(lines):
            s = line.strip()
            if s and any(x in s for x in ("Error", "Exception", "TypeError")):
                error_type = s
                break
        parts = []
        if error_type:
            parts.append(f"Error: {error_type}")
        if user_lines:
            parts.append("In your scene code:")
            parts.extend(f"  {l.strip()}" for l in user_lines[-4:])
        parts.append("\nFull traceback (last portion):")
        parts.append(error[-600:])
        return "\n".join(parts)