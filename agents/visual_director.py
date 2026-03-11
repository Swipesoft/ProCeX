"""
agents/visual_director.py

THE most important agent in the pipeline.

For each scene, given the narration + word timestamps + domain skill pack,
the VisualDirector makes THREE decisions:
  1. visual_strategy: MANIM | IMAGE_GEN | IMAGE_MANIM_HYBRID | TEXT_ANIMATION
  2. visual_prompt: detailed, ready-to-execute prompt for ManimCoder or ImageGenAgent
  3. needs_labels / label_list: if IMAGE_GEN, does it need callout labels?

The director is the only agent that reads the skill pack's image_gen_triggers
and manim_preferred_topics to make intelligent, context-aware decisions.

Key principle: when in doubt, use MANIM.
IMAGE_GEN is reserved for scenes that genuinely need a spatial/anatomical image
that Manim cannot reproduce — not just because it's "medical."
"""
from __future__ import annotations
import json
from state import ProcExState, Scene, VisualStrategy
from config import ProcExConfig, CINEMATIC_PALETTE
from utils.llm_client import LLMClient
from utils.timestamp_utils import timestamps_to_dict_list
from agents.base_agent import BaseAgent


DIRECTOR_SYSTEM = """You are a master animation director for cinematic educational videos.

For each scene, you decide the OPTIMAL visual strategy and write a precise visual brief.

VISUAL STRATEGIES:
─────────────────────────────────────────────────────────────────────────────
MANIM          → Python Manim animations. Use for:
                 • Mathematical equations, proofs, derivations
                 • Algorithm step-by-step walkthroughs
                 • Flowcharts, pathophysiology cascades, mechanism diagrams
                 • Data flow, architecture diagrams
                 • Lab value charts, drug comparison tables
                 • Clinical reasoning frameworks (BowTie, ADPIE, priority lists)
                 • Pharmacology MOA (receptor diagrams, enzyme cascades)
                 • ANYTHING conceptual/abstract that doesn't need a real image
                 • DEFAULT FOR ALL DOMAINS unless anatomy is explicitly needed

IMAGE_GEN      → NanoBanana (Gemini image generation). Use ONLY for:
                 • Scenes where the viewer MUST see a real anatomical structure spatially
                 • Gross anatomy (cross-sections, organ location, body regions)
                 • Histology (cell/tissue appearance the viewer needs to recognize)
                 • Physical assessment findings (wound, skin, inspection)
                 • When the script says things like "here you can see the [organ]"
                 DO NOT use for: mechanisms, pharmacology, pathophysiology, clinical reasoning

IMAGE_MANIM_HYBRID → NanoBanana background image + Manim text/arrows overlaid. Use for:
                 • Anatomy scene that also needs annotated arrows or process overlay
                 • "This is the kidney [image] — and here's how filtration works [Manim overlay]"

TEXT_ANIMATION → Manim but just cinematic text/title cards. Use for:
                 • Opening hooks, section transitions, closing synthesis
                 • Scenes that are narrative/conceptual with no specific visual anchor
─────────────────────────────────────────────────────────────────────────────

CRITICAL RULE: For medical/NCLEX content, MOST scenes use MANIM, not IMAGE_GEN.
Pathophysiology of diabetes → MANIM flowchart.
Pharmacology of opioids → MANIM receptor diagram.
Anatomy of the nephron → IMAGE_GEN (viewer needs to see the structure spatially).
This distinction is the most important judgment you make.

OUTPUT FORMAT: Return ONLY valid JSON array — no fences, no prose.
[
  {
    "scene_id": 1,
    "visual_strategy": "MANIM",
    "visual_reasoning": "This scene explains insulin resistance as a mechanism — Manim flowchart is ideal",
    "visual_prompt": "Detailed, precise prompt for the ManimCoder or ImageGenAgent...",
    "needs_labels": false,
    "label_list": [],
    "element_count": 4,
    "zone_allocation": {
      "scene_title":   "TITLE",
      "main_content":  "MAIN",
      "sidebar_note":  "UPPER_SIDEBAR",
      "footer_tip":    "FOOTER"
    }
  },
  ...
]

element_count: estimate the number of distinct visual elements (boxes, labels, icons,
arrows, text blocks) that will appear on screen simultaneously at peak density.
This is used by the layout critic to decide whether to inspect the rendered scene.
Be realistic — a simple title card is 1, a BowTie diagram with 5 arms is 7, a
medication table with 4 columns is 6.

zone_allocation: assign EVERY distinct on-screen element to a named zone from this list:
  TITLE        — top banner: scene title, chapter heading
  SUBTITLE     — second row: equation header, subtitle
  MAIN         — primary content: central diagram, proof, animation (rows 1–4, left 4 cols)
  UPPER_MAIN   — upper half of main: first half of a two-part layout
  LOWER_MAIN   — lower half of main: second half, answer reveal, detail
  SIDEBAR      — right sidebar: callout, legend, annotation (rows 1–4, right 2 cols)
  UPPER_SIDEBAR— upper sidebar: primary callout or highlighted fact
  LOWER_SIDEBAR— lower sidebar: secondary callout or follow-up note
  UPPER_LEFT   — upper-left quadrant: first panel in a 2×2 layout
  UPPER_RIGHT  — upper-right quadrant: second panel in a 2×2 layout
  LOWER_LEFT   — lower-left quadrant: third panel in 2×2
  LOWER_RIGHT  — lower-right quadrant: fourth panel in 2×2
  CENTER       — absolute canvas centre: isolated focus element
  FOOTER       — bottom banner: exam tip, recap line, citation

ZONE ALLOCATION RULES:
- Every key in zone_allocation must be a short snake_case label describing what the element IS
  (e.g. "formula", "step_counter", "priority_badge") — NOT a zone name
- No two elements may share the same zone. If you have more elements than zones,
  reduce element_count by grouping related elements into one composite element.
- TEXT_ANIMATION scenes: use only TITLE + optional SUBTITLE + optional FOOTER (max 3 zones)
- MANIM scenes: prefer TITLE + MAIN combination as the baseline; add SIDEBAR for secondary info
- A scene_title element MUST always occupy TITLE zone — every scene

For visual_prompt:
- MANIM: Describe exactly what Manim objects/animations to create. Reference specific Manim classes.
  Include timing guidance derived from the word timestamps. Be very specific.
- IMAGE_GEN: Write a complete NanoBanana prompt. Include style, resolution, content, labels needed.
  Format: "Medical illustration of [subject]. [Style]. [Specific structures to show]. [4K/2K]."
- IMAGE_MANIM_HYBRID: Describe both the background image AND the Manim overlay separately.
"""


def _build_director_prompt(state: ProcExState) -> str:
    skill = state.skill_pack
    triggers   = skill.get("image_gen_triggers", [])
    preferred  = skill.get("manim_preferred_topics", [])
    manim_style = skill.get("manim_style", "")
    image_style = skill.get("image_gen_style", "")
    manim_elements = skill.get("manim_elements", [])

    scenes_json = []
    for scene in state.scenes:
        ts_sample = timestamps_to_dict_list(scene.timestamps[:20])  # first 20 words
        scenes_json.append({
            "id":              scene.id,
            "narration_text":  scene.narration_text,
            "duration_seconds": round(scene.duration_seconds, 1),
            "initial_visual_hint": scene.visual_prompt,
            "word_timestamps_sample": ts_sample,
        })

    return f"""Domain: {state.domain.value}
Resolution: {state.resolution}

DOMAIN-SPECIFIC RULES:
Image gen ONLY triggers on scenes about: {triggers if triggers else "NOTHING — this domain never uses image gen"}
Manim preferred for: {preferred if preferred else "everything"}

MANIM STYLE GUIDE FOR THIS DOMAIN:
{manim_style}

IMAGE GEN STYLE GUIDE (only if using IMAGE_GEN):
{image_style}

AVAILABLE MANIM ELEMENTS FOR THIS DOMAIN:
{chr(10).join(f'  • {e}' for e in manim_elements)}

CINEMATIC PALETTE:
{json.dumps(CINEMATIC_PALETTE, indent=2)}

---
SCENES TO DIRECT:
{json.dumps(scenes_json, indent=2)}

---
Direct all {len(state.scenes)} scenes. Return the JSON array.
"""


class VisualDirector(BaseAgent):
    name = "VisualDirector"

    def run(self, state: ProcExState) -> ProcExState:
        self._log(f"Directing visuals for {len(state.scenes)} scenes...")

        for attempt in range(self.cfg.max_llm_retries):
            try:
                results = self.llm.complete_json(
                    DIRECTOR_SYSTEM,
                    _build_director_prompt(state),
                    max_tokens=16384,
                    temperature=0.4,
                    primary_provider="gemini",
                )
                break
            except Exception as e:
                self._log(f"Attempt {attempt+1} failed: {e}")
                if attempt == self.cfg.max_llm_retries - 1:
                    raise

        if not isinstance(results, list):
            results = results.get("scenes", []) if isinstance(results, dict) else []

        # Build lookup by scene_id
        by_id = {r["scene_id"]: r for r in results if "scene_id" in r}

        for scene in state.scenes:
            d = by_id.get(scene.id)
            if not d:
                self._log(f"No direction for scene {scene.id} — defaulting to MANIM")
                scene.visual_strategy = VisualStrategy.MANIM
                continue

            # Map string → enum
            strat_str = d.get("visual_strategy", "MANIM").upper()
            try:
                scene.visual_strategy = VisualStrategy(strat_str)
            except ValueError:
                self._log(f"Unknown strategy '{strat_str}' for scene {scene.id} → MANIM")
                scene.visual_strategy = VisualStrategy.MANIM

            scene.visual_reasoning  = d.get("visual_reasoning", "")
            scene.visual_prompt     = d.get("visual_prompt", "")
            scene.needs_labels      = d.get("needs_labels", False)
            scene.label_list        = d.get("label_list", [])
            scene.element_count     = int(d.get("element_count", 0))
            scene.zone_allocation   = d.get("zone_allocation", {})

            self._log(
                f"Scene {scene.id}: {scene.visual_strategy.value} | "
                f"elements={scene.element_count} | "
                f"zones={list(scene.zone_allocation.values())} | "
                f"{scene.visual_reasoning[:60]}"
            )

        # Strategy summary
        from collections import Counter
        counts = Counter(s.visual_strategy.value for s in state.scenes)
        self._log(f"Strategy distribution: {dict(counts)}")

        return state