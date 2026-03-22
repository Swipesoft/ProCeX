"""
agents/image_reprompter.py

ImageReprompter — middleware agent that transforms a sparse visual_prompt into
a cinematically rich, hyper-specific image generation directive.

Called when VLMCritic flags a scene for imagegen_fallback (split_depth >= 1).
The reprompter receives the full video context — all scenes, the script, the
style, the domain — and produces a prompt so detailed that the image generator
can execute it pixel-perfectly without ambiguity.

The output prompt is injected directly into scene.visual_prompt before the
ImageGenAgent runs. The image generator receives no reference to its internal product name.
"""
from __future__ import annotations
from agents.base_agent import BaseAgent


REPROMPTER_SYSTEM = """\
You are a world-class art director and cinematographer specialising in
educational content for social media. You receive a scene brief from a
video pipeline and produce one hyper-specific image generation directive.

Your output will be fed directly to a state-of-the-art text-to-image model
capable of rendering photorealistic scenes, stylised anime, infographics,
data visualisations, typographic art, and cinematic illustrations — all with
professional precision. The more specific and vivid your directive, the better
the output.

YOUR JOB:
Transform the sparse scene brief into a fully art-directed prompt that specifies:
  1. SCENE TYPE — choose the single most effective visual format:
       • Cinematic portrait (scientist, figure, character)
       • Anime / manga illustration
       • Infographic / diagram
       • Data visualisation (chart, graph, timeline)
       • Abstract mathematical art
       • Architectural / historical scene
       • Symbolic / metaphorical composition
       • Split-panel comparison
  2. SUBJECT — who or what is the central focus? If a famous person, describe
     their appearance, clothing, expression, and body language in detail.
     If an equation or diagram, describe its visual form precisely.
  3. SETTING — background, environment, era, lighting, atmosphere.
  4. TYPOGRAPHY — if text appears in the image, specify font style (serif,
     sans-serif, handwritten, chalk, neon, carved stone), size hierarchy,
     placement, and colour.
  5. COLOUR PALETTE — dominant hues, accent colours, contrast level.
     Default palette: deep space black (#0A0A0F background), electric cyan
     (#00D4FF) and purple (#7B2FFF) accents — but override when the scene
     demands it (e.g. ancient papyrus, whiteboard, golden age mathematics).
  6. COMPOSITION — where is the subject? What occupies foreground, midground,
     background? Rule of thirds, symmetry, dynamic diagonal?
  7. STYLE REFERENCE — e.g. "Studio Ghibli background painting", "NASA
     technical illustration", "Soviet constructivist poster", "3Blue1Brown
     visual style", "Renaissance oil painting chiaroscuro".
  8. TECHNICAL SPECS — aspect ratio, resolution, safe zones (if TikTok).
  9. WHAT TO AVOID — explicitly state anything that should NOT appear.

CRITICAL RULES:
  • Never include brand names of image generation tools in the prompt.
  • Never say "generate", "create", "draw", "make", "render", or "produce".
    Write the prompt as a description of the image that ALREADY EXISTS.
  • Write in present tense, third person: "Einstein stands at a blackboard..."
  • Be specific about numbers: "seven glowing nodes", "three parallel lines",
    not "several nodes", "some lines".
  • Famous historical figures may be named and described visually.
  • The prompt must feel like a director briefing a cinematographer —
    confident, specific, evocative, leaving nothing to chance.

OUTPUT FORMAT:
Return ONLY valid JSON. No preamble, no markdown fences.
{
  "scene_type": "<one of the 8 scene types above>",
  "enriched_prompt": "<the complete hyper-specific image prompt, 150-400 words>",
  "negative_prompt": "<things to exclude, comma-separated>",
  "style_reference": "<concise style label, e.g. '3Blue1Brown cinematic'>"
}
"""


REPROMPTER_USER_TMPL = """\
VIDEO CONTEXT
=============
Topic:            {topic}
Domain:           {domain}
Presentation style: {style}
Total scenes:     {total_scenes}
Video arc:        {video_arc}

FULL SCRIPT (all narration, for mood reference):
{full_script}

NEIGHBOURING SCENES (for visual continuity):
{neighbours}

THIS SCENE TO REPROMPT
======================
Scene ID:         {scene_id}
Position:         Scene {position} of {total_scenes}
Narration:        {narration}
Current prompt:   {current_prompt}
Duration:         {duration:.1f}s
Aspect ratio:     {aspect_ratio}
{tiktok_note}

TASK
====
This scene will be rendered as a static image (with a subtle camera drift
animation applied in post). The image must:
  1. Visually represent the narration text in the most impactful, memorable way.
  2. Feel like it belongs in the same cinematic sequence as its neighbours.
  3. Carry the emotional register of the presentation style ({style}).
  4. Be immediately legible at {aspect_ratio} on a mobile screen.

Produce the hyper-specific image directive now.
"""


class ImageReprompter(BaseAgent):
    """
    Middleware agent between VLMCritic's imagegen_fallback decision and
    the ImageGenAgent's prompt execution.

    Takes the sparse visual_prompt + full video context → returns the scene
    with scene.visual_prompt replaced by a hyper-specific cinematic directive.
    """
    name = "ImageReprompter"

    def reprompt(self, scene, state) -> object:
        """
        Enrich scene.visual_prompt with a SOTA image generation directive.
        Returns the scene object with visual_prompt updated in place.
        """
        from config import RESOLUTIONS

        res         = RESOLUTIONS.get(state.resolution, RESOLUTIONS["1080p"])
        aspect      = res.aspect_ratio
        is_portrait = res.is_portrait

        # ── Gather neighbouring scene context ────────────────────────────────
        all_ids  = [s.id for s in state.scenes]
        my_idx   = next((i for i, s in enumerate(state.scenes) if s.id == scene.id), -1)

        def _narr(idx: int) -> str:
            if 0 <= idx < len(state.scenes):
                return state.scenes[idx].narration_text[:200]
            return ""

        prev_scene  = state.scenes[my_idx - 1] if my_idx > 0 else None
        next_scene  = state.scenes[my_idx + 1] if my_idx < len(state.scenes) - 1 else None

        neighbour_block = ""
        if prev_scene:
            neighbour_block += (
                f"PREVIOUS (Scene {prev_scene.id}): {prev_scene.narration_text[:200]}\n"
                f"  Visual: {prev_scene.visual_prompt[:150]}\n\n"
            )
        if next_scene:
            neighbour_block += (
                f"NEXT (Scene {next_scene.id}): {next_scene.narration_text[:200]}\n"
                f"  Visual: {next_scene.visual_prompt[:150]}\n"
            )
        if not neighbour_block:
            neighbour_block = "This is a standalone scene."

        # ── Infer video arc from style pack ──────────────────────────────────
        style_name  = state.style_pack.get("display_name", state.presentation_style)
        narrative_arcs = state.style_pack.get("narrative_arcs", "")
        video_arc   = narrative_arcs[:300] if narrative_arcs else f"{style_name} presentation"

        # ── TikTok safe zone note ─────────────────────────────────────────────
        tiktok_note = ""
        if is_portrait:
            tiktok_note = (
                "SAFE ZONE (CRITICAL): This is a 9:16 TikTok portrait image.\n"
                "  Keep ALL subjects and text within the safe band:\n"
                "  • LEFT 40px (≈4% of width): EMPTY — physical screen edge clips text\n"
                "  • Top 130px (≈7% of height): EMPTY — TikTok search bar\n"
                "  • Bottom 200px (≈10% of height): EMPTY — caption/username bar\n"
                "  • Right 120px (≈11% of width): EMPTY — like/share/follow buttons\n"
                "  Compose with generous left breathing room — never anchor text to the left edge.\n"
                "  The safe content area is the centre 85% of the frame."
            )

        # ── Full script excerpt (mood reference, truncated) ───────────────────
        full_script = state.full_script_text[:1200] + ("..." if len(state.full_script_text) > 1200 else "")

        user = REPROMPTER_USER_TMPL.format(
            topic          = state.topic_slug.replace("_", " "),
            domain         = state.domain.value,
            style          = style_name,
            total_scenes   = len(state.scenes),
            video_arc      = video_arc,
            full_script    = full_script,
            neighbours     = neighbour_block,
            scene_id       = scene.id,
            position       = my_idx + 1,
            narration      = scene.narration_text,
            current_prompt = scene.visual_prompt[:400],
            duration       = getattr(scene, "tts_duration", scene.duration_seconds) or scene.duration_seconds,
            aspect_ratio   = aspect,
            tiktok_note    = tiktok_note,
        )

        try:
            result = self.llm.complete_json(
                REPROMPTER_SYSTEM, user,
                max_tokens       = 4096,
                temperature      = 0.85,   # creative but not chaotic
                primary_provider = "gemini",
            )

            enriched = result.get("enriched_prompt", "").strip()
            negative = result.get("negative_prompt", "").strip()
            style_ref= result.get("style_reference", "").strip()

            if len(enriched) < 40:
                self._log(f"Scene {scene.id}: reprompter returned thin prompt — keeping original")
                return scene

            # Compose final prompt: enriched + negative + style ref
            final_prompt = enriched
            if style_ref:
                final_prompt += f" Style: {style_ref}."
            if negative:
                final_prompt += f" Avoid: {negative}."

            scene.visual_prompt = final_prompt
            self._log(
                f"Scene {scene.id}: reprompter enriched prompt "
                f"({len(enriched)} chars, type={result.get('scene_type','?')})"
            )

        except Exception as e:
            self._log(f"Scene {scene.id}: reprompter failed ({e}) — keeping original prompt")

        return scene

    def run(self, state):
        """Not used in pipeline — called directly via reprompt()."""
        return state