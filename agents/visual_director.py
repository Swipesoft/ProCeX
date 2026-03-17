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
                 • ANYTHING conceptual/abstract that does not require a real image

IMAGE_GEN      → NanoBanana (Gemini image generation). MANDATORY for:
                 • Scenes where the viewer MUST spatially recognise a real
                   anatomical or physical structure they would see in a textbook
                 • Gross anatomy (organ location, cross-sections, body regions)
                 • Histology (cell/tissue appearance the viewer needs to recognise)
                 • Physical assessment findings (wound, skin, inspection findings)
                 • Any scene whose narration says or implies "here you can see
                   the [structure]" or "notice the appearance of [structure]"

IMAGE_MANIM_HYBRID → NanoBanana background + Manim overlay. MANDATORY for:
                 • Anatomy scene that also needs annotated arrows OR process overlay
                 • "This is the kidney [image] — here's how filtration works [overlay]"

TEXT_ANIMATION → Manim cinematic text/title cards only. Use for:
                 • Opening hooks, section transitions, closing synthesis
                 • Narrative/conceptual scenes with no specific visual anchor
─────────────────────────────────────────────────────────────────────────────

══════════════════════════════════════════════════════════════════
IMAGE_GEN TRIGGER GATE — READ THIS BEFORE DECIDING EVERY SCENE
══════════════════════════════════════════════════════════════════
You will be given a list of image_gen_triggers for the active domain.
These are NOT suggestions. They are MANDATORY conditions:

  IF the scene narration contains or strongly implies any trigger topic
  THEN visual_strategy MUST be IMAGE_GEN or IMAGE_MANIM_HYBRID.
  MANIM is NOT a valid choice for these scenes — Manim cannot render
  spatial anatomy, histology, or physical assessment findings.

The triggers define the boundary between "conceptual" (→ MANIM) and
"spatial/structural" (→ IMAGE_GEN). When in doubt about which side a
scene falls on, ask: "Would a medical textbook show a photograph or
diagram here?" If yes → IMAGE_GEN. If it would show a flowchart or
equation → MANIM.

Examples of correct decisions:
  "the glomerulus filters blood through fenestrated capillaries"
    → IMAGE_GEN  (viewer needs to SEE the glomerular structure)
  "insulin resistance occurs when receptors fail to respond"
    → MANIM      (mechanism/cascade — no spatial anatomy needed)
  "here we see the cross-section of the renal corpuscle"
    → IMAGE_GEN  (explicit spatial anatomy)
  "the coagulation cascade activates factor X"
    → MANIM      (pathway/flowchart — no real image needed)
  "podocytes form filtration slits along the basement membrane"
    → IMAGE_MANIM_HYBRID  (needs anatomy image + Manim overlay for slits)
══════════════════════════════════════════════════════════════════

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
- No two elements may share the same zone
- TEXT_ANIMATION scenes: use only TITLE + optional SUBTITLE + optional FOOTER
- A scene_title element MUST always occupy TITLE zone — every scene

For visual_prompt:
- MANIM: Describe exactly what Manim objects/animations to create. Be very specific.
- IMAGE_GEN: Write a complete NanoBanana prompt:
  "Medical illustration of [subject]. [Style]. [Specific structures]. [4K/2K]."
- IMAGE_MANIM_HYBRID: Describe the background image AND the Manim overlay separately.
"""


def _build_director_prompt(state: ProcExState) -> str:
    from config import RESOLUTIONS
    from utils.spatial_grid import get_zones

    skill = state.skill_pack
    triggers       = skill.get("image_gen_triggers", [])
    preferred      = skill.get("manim_preferred_topics", [])
    manim_style    = skill.get("manim_style", "")
    image_style    = skill.get("image_gen_style", "")
    manim_elements = skill.get("manim_elements", [])

    res    = RESOLUTIONS.get(state.resolution, RESOLUTIONS["1080p"])
    aspect = res.aspect_ratio
    zones  = get_zones(aspect)
    zone_names = list(zones.keys())

    # Format triggers as a hard gate checklist, not a soft list
    if triggers:
        trigger_lines = "\n".join(f"  • {t}" for t in triggers)
        trigger_block = (
            "MANDATORY IMAGE_GEN TRIGGERS FOR THIS DOMAIN\n"
            "If a scene's narration involves ANY of the following, visual_strategy\n"
            "MUST be IMAGE_GEN or IMAGE_MANIM_HYBRID — MANIM is forbidden:\n"
            f"{trigger_lines}"
        )
    else:
        trigger_block = "IMAGE_GEN TRIGGERS: none — this domain never uses image generation."

    if preferred:
        manim_block = "MANIM is mandatory for: " + ", ".join(preferred)
    else:
        manim_block = "MANIM is the default for all non-trigger scenes."

    # Portrait-specific zone guidance
    if aspect == "9:16":
        zone_layout_note = (
            "ASPECT: 9:16 PORTRAIT — tall narrow canvas (phone/Reels/Shorts format).\n"
            "Zone allocation MUST use vertical stacking. SIDEBAR zone does not exist.\n"
            "Preferred layout: TITLE → UPPER_MAIN → MAIN → LOWER_MAIN → FOOTER.\n"
            "For two-panel layouts use UPPER_HALF / LOWER_HALF, not LEFT/RIGHT."
        )
    else:
        zone_layout_note = (
            "ASPECT: 16:9 LANDSCAPE — wide canvas (standard video format).\n"
            "Zone allocation can use horizontal layouts: MAIN + SIDEBAR is the baseline.\n"
            "Preferred layout: TITLE + MAIN, with SIDEBAR for secondary info."
        )

    scenes_json = []
    for scene in state.scenes:
        ts_sample = timestamps_to_dict_list(scene.timestamps[:20])
        scenes_json.append({
            "id":                    scene.id,
            "narration_text":        scene.narration_text,
            "duration_seconds":      round(scene.duration_seconds, 1),
            "initial_visual_hint":   scene.visual_prompt,
            "word_timestamps_sample": ts_sample,
        })

    return f"""Domain: {state.domain.value}
Resolution: {state.resolution} ({aspect})

{trigger_block}

{manim_block}

{zone_layout_note}

MANIM STYLE GUIDE FOR THIS DOMAIN:
{manim_style}

IMAGE GEN STYLE GUIDE (only if using IMAGE_GEN or HYBRID):
{image_style}

AVAILABLE MANIM ELEMENTS FOR THIS DOMAIN:
{chr(10).join(f'  • {e}' for e in manim_elements)}

CINEMATIC PALETTE:
{json.dumps(CINEMATIC_PALETTE, indent=2)}

AVAILABLE ZONES FOR zone_allocation ({aspect}):
{chr(10).join(f'  {n}: {zones[n].description}' for n in zone_names)}

---
SCENES TO DIRECT:
{json.dumps(scenes_json, indent=2)}

---
Direct all {len(state.scenes)} scenes. Return the JSON array.
Apply the IMAGE_GEN trigger gate rigorously — check each scene's narration_text
against the trigger list before assigning MANIM.
"""


class VisualDirector(BaseAgent):
    name = "VisualDirector"

    def run(self, state: ProcExState) -> ProcExState:
        n_scenes = len(state.scenes)
        self._log(f"Directing visuals for {n_scenes} scenes...")

        results = []
        for attempt in range(self.cfg.max_llm_retries):
            try:
                raw = self.llm.complete_json(
                    DIRECTOR_SYSTEM,
                    _build_director_prompt(state),
                    max_tokens=32768,   # raised — large scene counts need room
                    temperature=0.4,
                    primary_provider="gemini",
                )

                # Unwrap response
                raw_type = type(raw).__name__
                if isinstance(raw, list):
                    results = raw
                elif isinstance(raw, dict):
                    for key in ("scenes", "scene_directions", "data", "results"):
                        if key in raw and isinstance(raw[key], list):
                            results = raw[key]
                            break
                    else:
                        if "visual_strategy" in raw:
                            results = [raw]

                self._log(
                    f"Attempt {attempt+1}: raw_type={raw_type}, "
                    f"parsed {len(results)}/{n_scenes} scene direction(s)"
                )

                # Partial response detection — retry if model cut off early
                if len(results) < n_scenes:
                    self._log(
                        f"Attempt {attempt+1}: PARTIAL — only {len(results)} of "
                        f"{n_scenes} scenes returned. Retrying..."
                    )
                    results = []
                    if attempt < self.cfg.max_llm_retries - 1:
                        continue

                break

            except Exception as e:
                self._log(f"Attempt {attempt+1} failed: {e}")
                if attempt == self.cfg.max_llm_retries - 1:
                    raise

        # Diagnostics
        if results:
            self._log(f"Response item keys: {list(results[0].keys())}")
            from collections import Counter
            raw_strategies = [r.get("visual_strategy","?") for r in results if isinstance(r,dict)]
            self._log(f"Raw LLM strategies: {dict(Counter(raw_strategies))}")

        # Build lookup — handle both "scene_id" (schema) and "id" (model sometimes echoes input key)
        by_id = {}
        for r in results:
            if not isinstance(r, dict):
                continue
            sid = r.get("scene_id") or r.get("id")
            if sid is not None:
                by_id[int(sid)] = r

        for scene in state.scenes:
            d = by_id.get(int(scene.id))
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

