"""
agents/visual_director.py

THE most important agent in the pipeline.

For each scene, given the narration + word timestamps + domain skill pack,
the VisualDirector makes THREE decisions:
  1. visual_strategy: MANIM | IMAGE_GEN | TEXT_ANIMATION
  2. visual_prompt: detailed, ready-to-execute prompt for ManimCoder or ImageGenAgent
  3. needs_labels / label_list: if IMAGE_GEN, does it need callout labels?

The director is the only agent that reads the skill pack's image_gen_enabled flag
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

For each scene you make two decisions:
  1. The VISUAL STRATEGY (how to render it)
  2. Whether to SPLIT it into subscenes (if it is long and covers mixed content)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISUAL STRATEGY DECISION — SEMANTIC REASONING MODEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For every scene, answer this question first:

  "Is what the viewer needs to SEE a SPATIAL/STRUCTURAL REALITY
   (a physical thing they must recognise — anatomy, clinical finding,
   real-world object) or a CONCEPTUAL/PROCESS REALITY (something that
   can be CONSTRUCTED — a mechanism, sequence, comparison, formula)?"

SPATIAL/STRUCTURAL REALITY → IMAGE_GEN
  • The viewer must recognise a real anatomical structure, tissue, organ
  • A clinical assessment finding (JVD, wound, skin changes)
  • A cross-section, histology slide, gross anatomy specimen
  • Any scene where a medical textbook would show a photograph or atlas image

CONCEPTUAL/PROCESS REALITY → MANIM
  • Pathophysiology cascade, mechanism of action, feedback loop
  • Pharmacology: drug → receptor → clinical effect chains
  • Clinical reasoning framework (BowTie, ADPIE, priority lists)
  • Lab value comparisons, drug tables, flowcharts
  • Any scene where a textbook would show a flowchart, diagram, or equation

TEXT_ANIMATION → DEPRECATED — do NOT use this strategy.
  TEXT_ANIMATION produces plain text slides that look unprofessional and
  frequently render as black screens. It is kept only as an internal fallback.

  INSTEAD: For closing synthesis, opening hooks, or conceptual statements,
  use MANIM with a minimal cinematic layout:
  • 1-2 large text elements with FadeIn/Write animations
  • A simple geometric accent (line, arrow, pulse circle)
  • Dark background with electric cyan or purple text
  • This always produces better output than TEXT_ANIMATION

IMPORTANT: If image_gen_enabled is false for this domain, ALWAYS use MANIM
regardless of content type. ML/Math and CS domains never use IMAGE_GEN.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUBSCENE SPLITTING — CINEMATIC BEAT SEQUENCING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Long scenes (duration_seconds > split_threshold) that cover MIXED content
(e.g., "first we see the structure, then we explain the mechanism") should
be split into SUBSCENES — sequential visual beats, each with their own
strategy and duration.

This preserves cinematic flow: the video alternates between static imagery
and animated Manim sequences, creating rhythm rather than one long static shot.

When to split:
  • Scene duration > split_threshold AND narration covers distinct visual phases
  • Example: 50s scene — "The glomerulus [IMAGE, 15s] filters blood through
    the basement membrane [MANIM, 20s] under Starling forces [MANIM, 15s]"

When NOT to split:
  • Scene is pure MANIM from start to finish
  • Scene is short (under split_threshold)
  • Content flows as one continuous thought

Subscene duration_fraction must sum to 1.0 across all subscenes.
Minimum subscene duration: 10 seconds.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT: Return ONLY valid JSON array — no fences, no prose.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[
  {
    "scene_id": 1,
    "visual_strategy": "MANIM",
    "visual_reasoning": "Insulin resistance is a conceptual mechanism — Manim flowchart ideal",
    "visual_prompt": "Detailed prompt...",
    "needs_labels": false,
    "label_list": [],
    "element_count": 4,
    "zone_allocation": {"scene_title": "TITLE", "main_diagram": "MAIN"},
    "subscenes": []
  },
  {
    "scene_id": 3,
    "visual_strategy": "MANIM",
    "visual_reasoning": "Long scene with mixed structural and process content — splitting into beats",
    "visual_prompt": "",
    "needs_labels": false,
    "label_list": [],
    "element_count": 0,
    "zone_allocation": {},
    "subscenes": [
      {
        "beat": 1,
        "visual_strategy": "IMAGE_GEN",
        "duration_fraction": 0.30,
        "visual_prompt": "Medical illustration of the renal corpuscle showing glomerulus, Bowman capsule, podocytes. Clean textbook style. 16:9. 2K.",
        "needs_labels": true,
        "label_list": ["Glomerulus", "Bowman capsule", "Podocytes"],
        "element_count": 1,
        "zone_allocation": {}
      },
      {
        "beat": 2,
        "visual_strategy": "MANIM",
        "duration_fraction": 0.40,
        "visual_prompt": "Animated flowchart showing filtration forces: hydrostatic vs oncotic pressure arrows building up in sequence",
        "needs_labels": false,
        "label_list": [],
        "element_count": 5,
        "zone_allocation": {"scene_title": "TITLE", "force_diagram": "MAIN", "label": "FOOTER"}
      },
      {
        "beat": 3,
        "visual_strategy": "MANIM",
        "duration_fraction": 0.30,
        "visual_prompt": "Net filtration pressure equation with animated MathTex build-up showing NFP = HP - OP",
        "needs_labels": false,
        "label_list": [],
        "element_count": 3,
        "zone_allocation": {"scene_title": "TITLE", "equation": "MAIN", "result": "FOOTER"}
      }
    ]
  }
]

FIELDS:
- scene_id: matches the input scene id
- visual_strategy: for scenes WITHOUT subscenes — the single strategy for the whole scene
- subscenes: empty [] if scene is not being split; populated list if splitting
- When subscenes is non-empty: visual_strategy/visual_prompt/zone_allocation at scene level
  are IGNORED — use the per-subscene fields instead
- duration_fraction: fraction of parent scene duration this beat occupies (must sum to 1.0)
- zone_allocation: required for MANIM subscenes, empty {} for IMAGE_GEN subscenes
- element_count: peak simultaneous on-screen elements for this beat
"""


def _build_director_prompt(state: ProcExState) -> str:
    from config import RESOLUTIONS
    from utils.spatial_grid import get_zones

    skill = state.skill_pack
    image_enabled = skill.get("image_gen_enabled", True)
    manim_style = skill.get("manim_style", "")
    image_style = skill.get("image_gen_style", "")
    manim_elements = skill.get("manim_elements", [])

    res = RESOLUTIONS.get(state.resolution, RESOLUTIONS["1080p"])
    aspect = res.aspect_ratio
    zones = get_zones(aspect)
    zone_names = list(zones.keys())

    # Image generation gate — domain-level hard disable
    if not image_enabled:
        image_gate = (
            "IMAGE_GEN GATE: DISABLED for this domain.\n"
            "ALWAYS use MANIM regardless of content type. Never output IMAGE_GEN.\n"
            "Never output subscenes with IMAGE_GEN beats."
        )
    else:
        image_gate = (
            "IMAGE_GEN GATE: ENABLED for this domain.\n"
            "Use the SPATIAL vs CONCEPTUAL semantic reasoning model from the system prompt.\n"
            "Ask yourself: would a textbook show a PHOTOGRAPH here (-> IMAGE_GEN) or a FLOWCHART/EQUATION (-> MANIM)?\n"
            "For subscene splitting: use it when a long scene has distinct structural + process phases."
        )

    # ── Documentary paragraph type strategy hints ────────────────────────────
    # When scenes carry a paragraph_type from the documentary agent, bias
    # strategy selection. Empty string = non-documentary scene = normal logic.
    para_type_guidance = (
        "DOCUMENTARY PARAGRAPH TYPE HINTS (applies when paragraph_type field is non-empty):\n"
        "  TECHNICAL  → MANIM strongly preferred. Equations, derivations, diagrams.\n"
        "               Never use IMAGE_GEN for a TECHNICAL paragraph.\n"
        "  VOICE      → IMAGE_GEN strongly preferred. A dramatic portrait or historical\n"
        "               scene depicting the named character in their era and context.\n"
        "               Use MANIM only if IMAGE_GEN is domain-disabled.\n"
        "  NARRATOR   → Minimal. A clean title card or single animated text element.\n"
        "               Prefer short MANIM scene (1-2 elements max). Never overcrowd.\n"
        "  STORY      → Use content to decide: person/place → IMAGE_GEN, \n"
        "               process/concept → MANIM.\n"
        "  (empty)    → Normal pipeline: use standard SPATIAL vs CONCEPTUAL logic.\n"
        "NOTE: paragraph_type is a HINT, not an override. Domain image_gen gate still\n"
        "applies — if IMAGE_GEN is disabled for this domain, always use MANIM."
    )

    # Portrait vs landscape zone layout guidance
    if aspect == "9:16":
        zone_layout_note = (
            "ASPECT: 9:16 PORTRAIT -- tall narrow canvas.\n"
            "Stack content VERTICALLY. SIDEBAR does not exist. Use UPPER_MAIN/LOWER_MAIN/UPPER_HALF/LOWER_HALF."
        )
    else:
        zone_layout_note = (
            "ASPECT: 16:9 LANDSCAPE -- wide canvas.\n"
            "Use MAIN + SIDEBAR as baseline. Quadrant layouts available."
        )

    split_threshold = state.skill_pack.get("_cfg_split_threshold", 40.0)

    scenes_json = []
    for scene in state.scenes:
        ts_sample = timestamps_to_dict_list(scene.timestamps[:20])
        scenes_json.append({
            "id": scene.id,
            "narration_text": scene.narration_text,
            "duration_seconds": round(scene.duration_seconds, 1),
            "splittable": scene.duration_seconds > split_threshold,
            "initial_visual_hint": scene.visual_prompt,
            "word_timestamps_sample": ts_sample,
            "paragraph_type": getattr(scene, "paragraph_type", ""),
        })

    return f"""Domain: {state.domain.value}
Resolution: {state.resolution} ({aspect})
Subscene split threshold: {split_threshold}s (scenes marked splittable=true may use subscenes)

{image_gate}

{para_type_guidance}

{zone_layout_note}

MANIM STYLE GUIDE FOR THIS DOMAIN:
{manim_style}

IMAGE GEN STYLE GUIDE (only if image_gen_enabled):
{image_style}

AVAILABLE MANIM ELEMENTS FOR THIS DOMAIN:
{chr(10).join(f"  - {e}" for e in manim_elements)}

CINEMATIC PALETTE:
{json.dumps(CINEMATIC_PALETTE, indent=2)}

AVAILABLE ZONES FOR zone_allocation ({aspect}):
{chr(10).join(f"  {n}: {zones[n].description}" for n in zone_names)}

---
SCENES TO DIRECT:
{json.dumps(scenes_json, indent=2)}

---
Direct ALL {len(state.scenes)} scenes. Return the complete JSON array.
For each scene: apply semantic reasoning, use subscenes where splittable=true and content is mixed.
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
                    max_tokens=32768,  # raised — large scene counts need room
                    temperature=0.4,
                    primary_provider="gemini",
                )

                # Unwrap response
                raw_type = type(raw).__name__

                # ── DIAGNOSTIC: log raw response so we can see what model actually returned ──
                raw_str = str(raw)
                self._log(f"Attempt {attempt + 1}: raw_type={raw_type}, raw_length={len(raw_str)} chars")
                self._log(
                    f"Attempt {attempt + 1}: raw_keys={list(raw.keys()) if isinstance(raw, dict) else 'N/A (list)'}")
                self._log(f"Attempt {attempt + 1}: raw_preview={raw_str[:500]}")

                if isinstance(raw, list):
                    results = raw
                elif isinstance(raw, dict):
                    for key in ("scenes", "scene_directions", "data", "results"):
                        if key in raw and isinstance(raw[key], list):
                            results = raw[key]
                            self._log(f"Attempt {attempt + 1}: unwrapped via key='{key}', got {len(results)} items")
                            break
                    else:
                        if "visual_strategy" in raw:
                            results = [raw]
                            self._log(f"Attempt {attempt + 1}: single scene dict wrapped as list")
                        else:
                            self._log(
                                f"Attempt {attempt + 1}: UNRECOGNISED dict structure — full keys: {list(raw.keys())}")

                self._log(
                    f"Attempt {attempt + 1}: raw_type={raw_type}, "
                    f"parsed {len(results)}/{n_scenes} scene direction(s)"
                )

                # Partial response detection — retry if model cut off early
                if len(results) < n_scenes:
                    self._log(
                        f"Attempt {attempt + 1}: PARTIAL — only {len(results)} of "
                        f"{n_scenes} scenes returned. Retrying..."
                    )
                    results = []
                    if attempt < self.cfg.max_llm_retries - 1:
                        continue

                break

            except Exception as e:
                self._log(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.cfg.max_llm_retries - 1:
                    raise

        # Diagnostics
        if results:
            self._log(f"Response item keys: {list(results[0].keys())}")
            from collections import Counter
            raw_strategies = [r.get("visual_strategy", "?") for r in results if isinstance(r, dict)]
            self._log(f"Raw LLM strategies: {dict(Counter(raw_strategies))}")

        # Build lookup — handle both "scene_id" (schema) and "id" (model sometimes echoes input key)
        by_id = {}
        for r in results:
            if not isinstance(r, dict):
                continue
            sid = r.get("scene_id") or r.get("id")
            if sid is not None:
                by_id[int(sid)] = r

        # Track which scenes have subscene splits (expanded after loop)
        scenes_with_subscenes: list[tuple] = []  # (scene, subscene_dicts)

        for scene in state.scenes:
            d = by_id.get(int(scene.id))
            if not d:
                self._log(f"No direction for scene {scene.id} — defaulting to MANIM")
                scene.visual_strategy = VisualStrategy.MANIM
                continue

            # Check for subscene split first
            subscene_list = d.get("subscenes", [])
            if subscene_list and isinstance(subscene_list, list) and len(subscene_list) > 1:
                # Validate fractions sum to ~1.0
                total_frac = sum(float(b.get("duration_fraction", 0)) for b in subscene_list)
                if 0.85 <= total_frac <= 1.15:
                    self._log(
                        f"Scene {scene.id}: SUBSCENE SPLIT into {len(subscene_list)} beats "
                        f"({scene.duration_seconds:.0f}s total)"
                    )
                    scenes_with_subscenes.append((scene, subscene_list))
                    continue  # will be expanded below; skip single-scene assignment

            # Single strategy assignment
            strat_str = d.get("visual_strategy", "MANIM").upper()
            try:
                scene.visual_strategy = VisualStrategy(strat_str)
            except ValueError:
                self._log(f"Unknown strategy '{strat_str}' for scene {scene.id} → MANIM")
                scene.visual_strategy = VisualStrategy.MANIM

            scene.visual_reasoning = d.get("visual_reasoning", "")
            scene.visual_prompt = d.get("visual_prompt", "")
            scene.needs_labels = d.get("needs_labels", False)
            scene.label_list = d.get("label_list", [])
            scene.element_count = int(d.get("element_count", 0))
            scene.zone_allocation = d.get("zone_allocation", {})

            self._log(
                f"Scene {scene.id}: {scene.visual_strategy.value} | "
                f"elements={scene.element_count} | "
                f"zones={list(scene.zone_allocation.values())} | "
                f"{scene.visual_reasoning[:60]}"
            )

        # ── Expand subscene splits ─────────────────────────────────────────────
        if scenes_with_subscenes:
            state.scenes = self._expand_subscenes(state.scenes, scenes_with_subscenes)

        # Strategy summary
        from collections import Counter
        counts = Counter(s.visual_strategy.value for s in state.scenes)
        n_subscenes = sum(1 for s in state.scenes if s.subscene_index > 0)
        self._log(
            f"Strategy distribution: {dict(counts)}"
            + (f" ({n_subscenes} subscene beats)" if n_subscenes else "")
        )

        return state

    # ── Critic reroute ────────────────────────────────────────────────────────

    def reroute_scene(
            self,
            scene,
            frame_bytes: bytes,
            aspect: str = "16:9",
            force_split: bool = False,
    ):
        """
        Called by parallel_runner when VLMCritic returns status="reroute"
        or status="split_needed".

        force_split=True: used when split_needed fires (reroute budget exhausted).
        VisualDirector is told PATH B is mandatory — repositioning has already
        been tried twice and failed. It only decides HOW to split.

        Separation of concern: this method receives NO issue list from the
        Critic. It sees only the frame and the original scene intent. This
        means all reroute policy lives in the Critic; VisualDirector focuses
        purely on re-planning from what it sees.
        """
        import base64
        from utils.spatial_grid import get_zones

        zones = get_zones(aspect)
        zone_list = ", ".join(zones.keys())
        b64_frame = base64.b64encode(frame_bytes).decode("utf-8")

        if force_split:
            system_prompt = """\
You are an expert visual layout designer for educational animation videos.
You will receive a frame from a Manim animation and its original visual intent.
Repositioning elements has already been attempted twice and failed — the scene
is structurally too dense for a single frame.

YOUR ONLY TASK IS PATH B — SPLIT INTO SUBSCENES:
  Divide the content into 2-3 sequential beats, each covering ONE distinct
  teaching point with its own clean, uncluttered layout.
  Do NOT suggest revising the layout (PATH A) — that has already been tried.

Respond ONLY with valid JSON — no preamble, no markdown fences."""
        else:
            system_prompt = """\
You are an expert visual layout designer for educational animation videos.
You will receive a frame from a Manim animation and its original visual intent.
Your job is to decide ONE of two remediation paths:

PATH A — REVISE LAYOUT:
  The scene has a good amount of content but elements are badly positioned.
  Produce a new visual_prompt with explicit zone placements that eliminate
  the overlap. Keep it as one scene.

PATH B — SPLIT INTO SUBSCENES:
  The scene has too many teaching points to fit cleanly in one frame even
  with perfect positioning. Split the content into 2-3 sequential beats,
  each covering a distinct teaching point with its own clean layout.

Respond ONLY with valid JSON — no preamble, no markdown fences."""

        task_instruction = (
            "You MUST use PATH B — split into subscenes. Do not choose PATH A.\n"
            "Decide only HOW to split: what content goes in each beat."
            if force_split else
            "Look at the rendered frame carefully.\nDecide: PATH A (revise layout) or PATH B (split into subscenes)?"
        )

        user_prompt = f"""ORIGINAL SCENE INTENT
=====================
Scene ID:        {scene.id}
Visual strategy: {scene.visual_strategy.value}
Visual prompt:   {scene.visual_prompt}
Visual reasoning:{scene.visual_reasoning}
Element count:   {getattr(scene, 'element_count', '?')}
Zone allocation: {getattr(scene, 'zone_allocation', {})}

AVAILABLE ZONES: {zone_list}

RENDERED FRAME (peak density at ~90% of clip duration)
=======================================================
[Image attached — this is what actually rendered]

TASK
====
{task_instruction}

Respond with this JSON schema:

For PATH A:
{{
  "path": "A",
  "visual_prompt": "<new detailed visual_prompt with explicit zone names>",
  "zone_allocation": {{"element_name": "ZONE_NAME", ...}},
  "element_count": <int>,
  "reasoning": "<one sentence>"
}}

For PATH B:
{{
  "path": "B",
  "reasoning": "<one sentence why split is necessary>",
  "subscenes": [
    {{
      "beat": 1,
      "visual_prompt": "<what this beat shows>",
      "zone_allocation": {{"element_name": "ZONE_NAME"}},
      "element_count": <int>,
      "duration_weight": <0.0-1.0, must sum to 1.0>
    }}
  ]
}}"""

        try:
            raw = self.llm.complete_vision(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_bytes=frame_bytes,
                image_mime="image/jpeg",
                max_tokens=16384,
                temperature=0.3,
                primary_provider="gemini",
            )
            import re, json
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
            raw = re.sub(r"```\s*$", "", raw.strip(), flags=re.MULTILINE)
            raw = raw.strip()

            # Use json-repair to handle all malformed JSON cases
            from json_repair import repair_json
            try:
                result = json.loads(repair_json(raw, ensure_ascii=False))
                if not isinstance(result, dict):
                    raise ValueError(f"Expected dict, got {type(result).__name__}")
                self._log(f"[reroute_scene] Scene {scene.id}: JSON parsed (path={result.get('path', '?')})")
            except Exception as parse_err:
                raise ValueError(f"JSON unrecoverable: {parse_err}")
        except Exception as e:
            self._log(f"[reroute_scene] Scene {scene.id}: LLM call failed ({e}) — returning original")
            return scene

        path = result.get("path", "A")
        self._log(
            f"[reroute_scene] Scene {scene.id}: path={path} — {result.get('reasoning', '')}"
        )

        import dataclasses
        updated = dataclasses.replace(scene)

        if path == "A":
            updated = dataclasses.replace(
                updated,
                visual_prompt=result.get("visual_prompt", scene.visual_prompt),
                zone_allocation=result.get("zone_allocation", scene.zone_allocation),
                element_count=result.get("element_count", scene.element_count),
                visual_reasoning=f"[rerouted] {result.get('reasoning', '')}",
            )
            return updated

        else:  # PATH B — subscene split
            beats = result.get("subscenes", [])
            if not beats:
                self._log(
                    f"[reroute_scene] Scene {scene.id}: PATH B but no subscenes returned — falling back to PATH A no-op")
                return scene
            # Annotate the scene with new subscene beats so _expand_subscenes
            # can inflate them in the next parallel_runner cycle
            updated = dataclasses.replace(
                updated,
                visual_reasoning=f"[rerouted+split] {result.get('reasoning', '')}",
            )
            # Attach beats directly to scene object for the runner to expand
            object.__setattr__(updated, "_reroute_beats", beats)
            return updated

    # ── Subscene expansion ────────────────────────────────────────────────────

    def _expand_subscenes(
            self,
            scenes: list,
            splits: list[tuple],
    ) -> list:
        """
        Replace parent scenes with their subscene beats.
        Each beat becomes a full Scene with proportional duration and own strategy.
        Timestamps are sliced proportionally across beats.
        """
        from state import Scene as SceneClass

        # Build a map of parent id -> (parent_scene, [beat_dicts])
        split_map = {parent.id: (parent, beats) for parent, beats in splits}

        new_scenes = []
        # Use a global sub-id counter to keep ids unique
        # Subscene ids: parent_id * 100 + beat_index (e.g. scene 3 beat 2 → id 302)
        for scene in scenes:
            if scene.id not in split_map:
                new_scenes.append(scene)
                continue

            parent, beat_dicts = split_map[scene.id]

            # ── Audio-authoritative duration ──────────────────────────────────
            # Use tts_duration (actual recorded audio length) as the total to
            # distribute across beats. duration_seconds is the LLM estimate and
            # is often longer than the real audio, causing subscene audio offsets
            # to seek past the end of the file → silence in the final video.
            #
            # Rule: sum(beat_dur) MUST equal tts_audio_total, never exceed it.
            tts_audio_total = float(getattr(parent, "tts_duration", 0.0) or 0.0)
            total_dur = tts_audio_total if tts_audio_total > 1.0 else parent.duration_seconds
            all_timestamps = list(parent.timestamps)
            ts_count = len(all_timestamps)

            # Calculate raw beat durations from fractions, then clamp:
            # 1. Apply proportional split based on duration_fraction
            # 2. Apply 5s visual minimum (lower than 10s to avoid blowing past audio)
            # 3. If the sum exceeds total_dur, scale ALL beats down proportionally
            #    so they always fit within the actual recorded audio.
            n_beats = len(beat_dicts)
            raw_durs = []
            for beat in beat_dicts:
                frac = float(beat.get("duration_fraction", 1.0 / n_beats))
                raw_durs.append(max(round(total_dur * frac, 2), 5.0))

            # Rescale if sum exceeds real audio (prevents silence at end of last beat)
            raw_sum = sum(raw_durs)
            if raw_sum > total_dur + 0.1:
                scale = total_dur / raw_sum
                raw_durs = [round(d * scale, 2) for d in raw_durs]
                # Absorb rounding remainder into last beat
                remainder = round(total_dur - sum(raw_durs), 2)
                raw_durs[-1] = round(raw_durs[-1] + remainder, 2)

            # cursor tracks position within parent's duration (for timestamp slicing).
            # audio_cursor tracks absolute position in the combined audio file.
            cursor = 0.0
            audio_cursor = float(getattr(parent, "tts_audio_start", 0.0))

            for i, beat in enumerate(beat_dicts):
                beat_dur = raw_durs[i]

                # Slice timestamps proportionally within parent's range
                ts_start = int(cursor / total_dur * ts_count)
                ts_end = int((cursor + beat_dur) / total_dur * ts_count)
                beat_ts = all_timestamps[ts_start:ts_end]

                # Parse strategy
                strat_str = beat.get("visual_strategy", "MANIM").upper()
                try:
                    strategy = VisualStrategy(strat_str)
                except ValueError:
                    strategy = VisualStrategy.MANIM

                sub_id = parent.id * 100 + (i + 1)
                sub = SceneClass(
                    id=sub_id,
                    narration_text=parent.narration_text,
                    duration_seconds=beat_dur,
                    visual_strategy=strategy,
                    visual_prompt=beat.get("visual_prompt", ""),
                    visual_reasoning=f"Subscene {i + 1}/{len(beat_dicts)} of scene {parent.id}",
                    needs_labels=beat.get("needs_labels", False),
                    label_list=beat.get("label_list", []),
                    element_count=int(beat.get("element_count", 0)),
                    zone_allocation=beat.get("zone_allocation", {}),
                    timestamps=beat_ts,
                    tts_audio_path=parent.tts_audio_path,
                    tts_duration=beat_dur,
                    tts_audio_start=round(audio_cursor, 3),  # absolute offset in combined audio
                    parent_scene_id=parent.id,
                    subscene_index=i + 1,
                    split_depth=getattr(parent, "split_depth", 0) + 1,
                )
                new_scenes.append(sub)
                self._log(
                    f"  Beat {i + 1}/{len(beat_dicts)}: id={sub_id} "
                    f"{strategy.value} {beat_dur:.1f}s "
                    f"audio_start={round(audio_cursor, 3)}s"
                )
                cursor += beat_dur
                audio_cursor += beat_dur

        return new_scenes