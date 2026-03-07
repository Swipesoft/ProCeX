"""
agents/manim_coder.py
Generates runnable Manim Community Python code for MANIM / TEXT_ANIMATION scenes.

Validation pipeline (3 attempts before fallback):
  1. LLM generates class code
  2. _clean_code()        - strip fences, remove duplicate imports
  3. _validate_structure() - class name + construct method present
  4. _syntax_check()       - py_compile on full assembled file
  5. Fallback              - guaranteed-runnable cinematic title card

Render-error feedback: RendererAgent calls regenerate_for_render_error() when
Manim crashes at runtime. The exact traceback is injected into the next prompt
so the LLM self-corrects rather than regenerating blindly.
"""
from __future__ import annotations
import os
import re
import py_compile
import tempfile
from state import ProcExState, Scene, VisualStrategy
from config import ProcExConfig, MANIM_PALETTE_BLOCK
from utils.llm_client import LLMClient
from utils.timestamp_utils import timestamps_to_dict_list
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
You write cinematic, precisely-timed Manim animations for educational videos.

CORE RULES
==========
1. Output ONLY the Python class body — no imports, no config lines (already in file).
2. Class name: exactly as specified in scene_class_name.
3. Inherit from Scene: class SceneN(Scene):
4. def construct(self): is the only method.
5. First line of construct: self.camera.background_color = BG
6. Palette constants: BG, WHITE, CYAN, PURPLE, ORANGE, GREEN, GOLD, DIM
   These are ManimColor objects — pass them directly to color=, fill_color=, etc.
7. All animation timings must derive from word_timestamps.
   Total self.wait() + self.play(run_time=...) must equal duration_seconds.
8. Always end with self.wait(0.5).
9. DO NOT use: ImageMobject, SVGMobject, ThreeDScene, Surface, external files.
10. Equations: always MathTex(r"...") not Tex, always raw strings.
11. No TODO, no "...", no placeholder comments, no unfinished methods.

VGROUP vs GROUP — CRITICAL
==========================
VGroup only accepts VMobject subclasses (Text, MathTex, Square, Circle, Line,
Arrow, Dot, etc.). It CANNOT contain base Mobject instances.

RULE: If you are grouping objects and are not 100% certain all are VMobjects,
use Group() instead of VGroup(). Group() accepts any Mobject.

CORRECT:   Group(text, arrow, dot).arrange(DOWN)
CORRECT:   VGroup(Text("a"), MathTex(r"x^2"), Square())
WRONG:     VGroup(some_generic_mobject)  <- crashes with TypeError

Do NOT pass Mobject, ParametricCurve (without fill), or NumberPlane directly
into VGroup. Use Group() when mixing types.

LAYOUT RULES — VIOLATIONS CAUSE VISIBLE BUGS
=============================================
L1. FRAME BOUNDS: Frame is 14.22 wide x 8 tall (units).
    Keep all objects within x in [-5.5, 5.5] and y in [-3.3, 3.3].
    Use .move_to(ORIGIN) when unsure of position.

L2. CLEAR BETWEEN SECTIONS: Before new content, fade out existing objects:
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.5)
    Never let more than 4 objects accumulate simultaneously.

L3. TEXT SIZES:
    - Titles:     font_size=44 max
    - Body text:  font_size=32
    - Captions:   font_size=24
    - Never exceed font_size=52
    - Text over 50 chars: Text("...", font_size=28, width=11) for word wrap.

L4. MATH: Scale MathTex with > 2 terms:
        eq = MathTex(r"...").scale(0.85)
    Multi-line: MathTex(r"line1", r"line2").arrange(DOWN, buff=0.3)

L5. LAYOUT: Use Group/VGroup + .arrange() for multi-element layouts.
    Never manually shift() more than 2 objects — use .arrange() instead.

L6. NO OVERLAP: FadeOut old objects before drawing new ones in the same area.
    Use Transform() to morph one object into another in place.

L7. COLORS: Palette constants are ManimColor objects — use directly:
        interpolate_color(BG, CYAN, 0.5)  <- CORRECT
        interpolate_color("#0A0A0F", CYAN, 0.5)  <- WRONG (string crashes)

ANIMATION STYLE
===============
- Entrances: FadeIn(obj, shift=UP*0.3) or Write()
- Exits: always explicit FadeOut()
- Camera (requires MovingCameraScene): self.camera.frame.animate.scale(0.85)
- Transform(old, new) replaces old with new in one smooth animation

OUTPUT: Return ONLY the class definition. No fences, no explanation.
"""


# ── Fallback: guaranteed-runnable title card ───────────────────────────────────
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


def _build_coder_prompt(scene: Scene, skill: dict, error_context: str = "") -> str:
    ts_json        = timestamps_to_dict_list(scene.timestamps)
    timing_guide   = skill.get("timing_guidance", "")
    manim_elements = skill.get("manim_elements", [])

    if scene.timestamps:
        mid_word = scene.timestamps[len(scene.timestamps) // 2]
        mid_time = round(
            mid_word.start - (scene.timestamps[0].start if scene.timestamps else 0), 2
        )
    else:
        mid_time = scene.duration_seconds / 2

    error_block = ""
    if error_context:
        # Trim to 1200 chars so we don't blow the context window
        trimmed = error_context[-1200:] if len(error_context) > 1200 else error_context
        error_block = f"""
PREVIOUS ATTEMPT FAILED — READ AND FIX THIS ERROR:
---------------------------------------------------
{trimmed}
---------------------------------------------------
Do NOT repeat the same mistake. Study the traceback, identify the exact line
that caused it, and rewrite that section correctly.
"""

    return f"""Generate Manim code for this scene.
{error_block}
scene_class_name: {scene.manim_class_name}
scene_id: {scene.id}
duration_seconds: {scene.duration_seconds:.1f}
mid_scene_time: {mid_time:.1f}s

NARRATION (what is being said):
{scene.narration_text}

VISUAL BRIEF (what to animate):
{scene.visual_prompt}

WORD TIMESTAMPS:
{ts_json[:40]}
{"... (truncated)" if len(ts_json) > 40 else ""}

DOMAIN ELEMENTS:
{chr(10).join(f"  - {e}" for e in manim_elements)}

TIMING GUIDANCE:
{timing_guide}

Total timing must sum to {scene.duration_seconds:.1f}s.
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
        Regenerate and rewrite the .py file for a scene that failed at render time.
        Injects the Manim runtime traceback so the LLM can fix the exact problem.
        Returns True if new code was written, False if fallback was used.
        """
        self._log(
            f"Scene {scene.id}: render failed — regenerating with error context..."
        )

        # Extract the most useful part of the traceback (last 1200 chars)
        error_summary = self._summarise_render_error(render_error)

        code = self._generate_scene_code(
            scene, skill_pack, initial_error=error_summary
        )

        is_fallback = "_fallback" in code or f"Scene {scene.id}" in code[:80]
        self._write_scene_file(scene.manim_file_path, code)

        if is_fallback:
            self._log(f"Scene {scene.id}: regeneration fell back to title card")
            return False

        self._log(f"Scene {scene.id}: regenerated successfully")
        return True

    # ── Code generation ────────────────────────────────────────────────────────

    def _generate_scene_code(
        self,
        scene: Scene,
        skill: dict,
        initial_error: str = "",
    ) -> str:
        """
        Generate Manim class code with retries.
        Each failed attempt's error is fed back into the next prompt.
        """
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
                )
                code = self._clean_code(raw, scene.manim_class_name)

                struct_ok, struct_err = self._validate_structure(code, scene.manim_class_name)
                if not struct_ok:
                    last_error = f"Structural validation failed: {struct_err}"
                    self._log(f"Scene {scene.id} attempt {attempt+1}: {last_error}")
                    continue

                syntax_ok, syntax_err = self._syntax_check(header, code)
                if not syntax_ok:
                    last_error = f"Python syntax error — fix this before resubmitting:\n{syntax_err}"
                    self._log(f"Scene {scene.id} attempt {attempt+1}: syntax error: {syntax_err[:80]}")
                    continue

                self._log(f"Scene {scene.id} attempt {attempt+1}: PASS")
                return code

            except Exception as e:
                last_error = str(e)
                self._log(f"Scene {scene.id} attempt {attempt+1} exception: {e}")

        self._log(f"Scene {scene.id}: all attempts failed — using fallback title card")
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
        lines, cleaned = raw.splitlines(), []
        skip = {
            "from manim import", "import manim", "import numpy",
            "from manim.utils", "config.background", "config.frame_rate",
        }
        for line in lines:
            if not any(line.strip().startswith(s) for s in skip):
                cleaned.append(line)
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
            return False, "too many '...' placeholders — code is incomplete"
        return True, ""

    @staticmethod
    def _syntax_check(header: str, code: str) -> tuple[bool, str]:
        """Run py_compile on the full assembled file."""
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
        """
        Extract the most actionable part of a Manim traceback.
        Strips Manim internal frames, keeps the user-code line + error type.
        """
        lines = error.splitlines()

        # Find lines from the user's scene file (scene_NN.py)
        user_lines = [l for l in lines if "scene_" in l and ".py" in l]

        # Find the actual error type line (last non-empty line usually)
        error_type = ""
        for line in reversed(lines):
            stripped = line.strip()
            if stripped and ("Error" in stripped or "Exception" in stripped or "TypeError" in stripped):
                error_type = stripped
                break

        # Build a clean, focused summary for the LLM
        summary_parts = []
        if error_type:
            summary_parts.append(f"Error: {error_type}")
        if user_lines:
            summary_parts.append("In your scene code:")
            summary_parts.extend(f"  {l.strip()}" for l in user_lines[-4:])

        # Always include the last 600 chars of the raw traceback as full context
        summary_parts.append("\nFull traceback (last portion):")
        summary_parts.append(error[-600:])

        return "\n".join(summary_parts)