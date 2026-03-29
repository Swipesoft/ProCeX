"""
agents/deep_documentary.py

DeepDocumentaryAgent — converts a bare topic into a Netflix-style documentary
PDF, using iterative Tavily web research to gather facts, then an LLM to
compose a multi-voice, cinematically structured document.

Output PDF is saved to papers/ and handed back to main.py as --input.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARAGRAPH TYPES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[NARRATOR]  — The anchor. Short transitional bridges (2-4 sentences).
              Introduces characters, sets up transitions, maintains tension.
              Appears before and after every B/C/D block.
              Voice: Aoede (female narrator). Never teaches. Never opines.
              Just moves the story forward and hands off to the next voice.

[STORY]     — Historical/biographical prose. The cinematic backstory.
              Who was this person, what era, what motivated them, what
              happened. Grounded in real dates and events. Longer paragraphs.
              Voice: Aoede. This is the "documentary narrator" in full flight.

[TECHNICAL] — The tutoring paragraph. Actual science/math explained
              accessibly. Uses analogies. Walks through derivations simply.
              Assumes an intelligent non-specialist reader.
              Voice: Aoede. ScriptWriter will produce MathTex-heavy scenes.

[VOICE: X]  — First-person critic/ally. X is a real named historical figure.
              Speaks in their documented voice — their actual quotes woven
              into plausible first-person reconstruction.
              Voice: Male Gemini TTS voice. One voice per named figure.
              Always sandwiched between [NARRATOR] paragraphs.

FLOW RULE:
  [NARRATOR] is the spine. Every B/C/D block must be preceded and followed
  by a [NARRATOR] paragraph. You can never jump directly from [TECHNICAL]
  to [VOICE] — the narrator always bridges them.

  Valid: N → S → N → T → N → V → N → T → N → V → N
  Invalid: S → T → V → T (narrator missing between each)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESEARCH LOOP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Max 15 Tavily searches. Each search is decided by the LLM after reading
all previous results — it follows the story wherever it leads (people,
controversies, dates, technical disputes). The LLM decides what is still
unknown and what would make the documentary richer. It does NOT follow a
pre-planned outline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stage 1 — Seed search         : first Tavily call to orient the topic
Stage 2 — Iterative research  : LLM-guided follow-up searches (max 14 more)
Stage 3 — Documentary writer  : LLM composes the full tagged document
Stage 4 — PDF assembler       : ReportLab renders the PDF with type styling
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from agents.base_agent import BaseAgent
from config import ProcExConfig
from utils.llm_client import LLMClient
from utils.slug import slugify


# ── Paragraph type constants ──────────────────────────────────────────────────
P_NARRATOR  = "NARRATOR"
P_STORY     = "STORY"
P_TECHNICAL = "TECHNICAL"
P_VOICE     = "VOICE"    # VOICE: Einstein, VOICE: Bohr, etc.

# ── Tavily search depth ───────────────────────────────────────────────────────
MAX_SEARCHES   = 15
RESULTS_PER_Q  = 5      # max_results per Tavily call
SEARCH_DEPTH   = "advanced"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM prompts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_NEXT_QUERY_SYSTEM = """\
You are a documentary researcher. Your ONLY job is to output a JSON object
specifying the next web search query to run.

You will receive: a topic, a summary of what has been found so far, and
a list of gaps still needing research.

You must respond with EXACTLY this JSON structure and nothing else:
{
  "done": false,
  "reason": "one sentence explaining what gap this search fills",
  "query": "3 to 8 word search phrase"
}

CONCRETE EXAMPLE — topic is Euclid, we know biography but not the controversy:
{
  "done": false,
  "reason": "Need to find who first challenged the parallel postulate and when",
  "query": "Lobachevsky parallel postulate challenge 1830"
}

EXAMPLE when story is complete:
{
  "done": true,
  "reason": "",
  "query": ""
}

RULES for the query field:
  - Minimum 3 words, maximum 8 words
  - A real searchable phrase, not a question
  - Include names, dates, or places where possible
  - Never repeat a query already searched

Output ONLY the JSON object. No explanation. No markdown. No preamble.
"""

_NEXT_QUERY_USER = """\
TOPIC: {topic}

SEARCHES COMPLETED SO FAR: {n_done} of {max_searches}

RESEARCH GATHERED SO FAR:
{summary}

FIGURES ALREADY COVERED: {characters}

WHAT IS STILL MISSING:
{gaps}

Now output your JSON response with the next search query.
Remember: respond with ONLY the JSON object, nothing else.
"""

_WRITER_SYSTEM = """\
You are the lead writer for a Netflix-style documentary series on science,
mathematics, and history. You write scripts that are cinematic, precise,
emotionally resonant, and scientifically accurate.

You write in a specific paragraph-tagging format. Each paragraph is tagged
with its type on the line before it:

  [NARRATOR]
  The narrator's voice. Short transitional bridges. 2-4 sentences only.
  Introduces the next character or idea. Creates tension. Never teaches.
  This is the SPINE of the documentary — it appears before and after every
  other paragraph type.

  [STORY]
  Biographical and historical prose. Who was this person? What was their
  world like? What drove them? Ground every claim in real dates and facts.
  Write cinematically — specific details, specific places, specific moments.

  [TECHNICAL]
  The tutoring paragraph. Explain the actual science or mathematics.
  Assume an intelligent non-specialist. Use analogies. Walk through the
  key equation or concept step by step. This is where the viewer learns.

  [VOICE: Name]
  First-person reconstruction. A named historical figure speaks directly
  to the camera. Weave in their documented real quotes where they exist.
  Write in their intellectual voice — their concerns, their doubts, their
  convictions. Passionate but grounded in documented history.

MANDATORY FLOW RULE:
  [NARRATOR] must appear before AND after every [STORY], [TECHNICAL], or
  [VOICE] paragraph. You cannot place two non-NARRATOR paragraphs adjacently.
  The narrator is the thread that stitches everything together.

  VALID:   NARRATOR → STORY → NARRATOR → TECHNICAL → NARRATOR → VOICE → NARRATOR
  INVALID: STORY → TECHNICAL → VOICE (narrator missing between each)

SUSPENSE RULE:
  The opening [NARRATOR] paragraph must open with a question or provocation
  that makes the viewer feel they cannot stop watching.
  ✓ "Was Schrödinger insane to think the universe was not made of particles,
     but of waves resolving themselves in real-time?"
  ✓ "For two thousand years, everyone agreed Euclid was right.
     Everyone, that is, except the mathematicians who were too afraid to say so."

SCIENTIFIC ACCURACY:
  Every date, name, and scientific claim must be grounded in the research
  material provided. Do not invent quotes — reconstruct plausibly from
  documented positions. Do not speculate about private thoughts.

OUTPUT FORMAT:
  Return ONLY valid JSON. No markdown fences. No preamble.
  {{
    "title": "<documentary title>",
    "tagline": "<one-line dramatic hook>",
    "suggested_minutes": <float, 5.0-8.0>,
    "suggested_style": "<tiktok-thriller or youtube-tutorial>",
    "characters": ["<name>", ...],
    "paragraphs": [
      {{
        "type": "NARRATOR",
        "speaker": null,
        "text": "..."
      }},
      {{
        "type": "STORY",
        "speaker": null,
        "text": "..."
      }},
      {{
        "type": "VOICE",
        "speaker": "Einstein",
        "text": "..."
      }}
    ]
  }}

PARAGRAPH COUNT GUIDANCE:
  Target {target_paragraphs} total paragraphs for a {target_minutes:.0f}-minute video.
  Every [NARRATOR] paragraph = ~1 scene. Every [STORY]/[TECHNICAL] = ~2 scenes.
  Every [VOICE] = ~1 scene. Keep individual paragraphs tight — 80-150 words each.
"""

_WRITER_USER = """\
DOCUMENTARY TOPIC: {topic}

TARGET LENGTH: {target_minutes:.0f} minutes ({target_paragraphs} paragraphs approx.)

ALL RESEARCH GATHERED ({n_searches} web searches):
{research_digest}

CHARACTERS/FIGURES IDENTIFIED:
{characters}

Write the full documentary script now. Follow the paragraph tagging format
exactly. Remember: NARRATOR is always the bridge between every other paragraph.
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data classes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class SearchResult:
    query:    str
    reason:   str
    snippets: list[dict]   # raw Tavily results[{title, url, content, score}]
    answer:   str          # Tavily's answer field (often empty)


@dataclass
class Paragraph:
    type:    str           # NARRATOR | STORY | TECHNICAL | VOICE
    speaker: Optional[str] # None except for VOICE paragraphs
    text:    str


@dataclass
class DocumentaryScript:
    title:             str
    tagline:           str
    topic:             str
    suggested_minutes: float
    suggested_style:   str
    characters:        list[str]
    paragraphs:        list[Paragraph]
    searches_used:     int


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DeepDocumentaryAgent(BaseAgent):
    """
    Researches a topic via iterative Tavily searches, then writes a
    Netflix-style multi-voice documentary PDF for the ProcEx pipeline.
    """
    name = "DeepDocumentary"

    def __init__(self, cfg: ProcExConfig, llm: LLMClient):
        super().__init__(cfg, llm)
        self._tavily = None   # lazy-init so missing key gives a clear error

    def _get_tavily(self):
        if self._tavily is not None:
            return self._tavily
        try:
            from tavily import TavilyClient
        except ImportError:
            raise RuntimeError(
                "tavily-python not installed. Run: pip install tavily-python"
            )
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY not set. Add it to your .env file."
            )
        self._tavily = TavilyClient(api_key=api_key)
        return self._tavily

    # ── Public entry point ────────────────────────────────────────────────────

    def research(
        self,
        topic:          str,
        target_minutes: float = 6.0,
        presentation_style: str = "tiktok-thriller"
    ) -> str:
        """
        Full pipeline: research → write → assemble PDF.
        Returns path to the generated PDF.
        """
        self._log(f"Starting deep documentary: '{topic}'")
        start = time.time()

        # Stage 1+2: Research
        searches = self._research_loop(topic)

        # Stage 3: Write
        script = self._write_documentary(topic, searches, target_minutes, presentation_style=presentation_style)

        # Stage 4: PDF
        pdf_path = self._build_pdf(script, topic)

        elapsed = time.time() - start
        self._log(
            f"Documentary complete in {elapsed/60:.1f} min — "
            f"{len(script.paragraphs)} paragraphs, "
            f"{script.searches_used} searches, "
            f"PDF: {pdf_path}"
        )
        return pdf_path

    # ── Stage 1+2: Iterative research loop ───────────────────────────────────

    def _research_loop(self, topic: str) -> list[SearchResult]:
        tavily   = self._get_tavily()
        searches: list[SearchResult] = []
        characters: list[str] = []

        self._log(f"Research loop started (max {MAX_SEARCHES} searches)")

        for i in range(MAX_SEARCHES):
            # Decide query
            if i == 0:
                # Seed: broad orientation
                query  = topic
                reason = "Initial orientation search"
                self._log(f"  Search 1/{MAX_SEARCHES}: '{query}' [seed]")
            else:
                # LLM decides next query based on what it found
                decision = self._decide_next_query(
                    topic      = topic,
                    searches   = searches,
                    characters = characters,
                    n_done     = i,
                )
                if decision.get("done", False):
                    self._log(
                        f"  LLM declared research complete after {i} searches"
                    )
                    break
                query  = decision.get("query", "").strip()
                reason = decision.get("reason", "")
                if not query or len(query) < 3:
                    self._log(f"  LLM returned invalid query {query!r} — skipping")
                    continue   # try next iteration rather than stopping entirely
                # Word count gate: reject queries that are clearly truncated
                word_count = len(query.split())
                if word_count < 2:
                    self._log(f"  Query too short ({word_count} words): {query!r} — skipping")
                    continue
                if word_count > 12:
                    # Truncate to first 8 words rather than discard entirely
                    query = " ".join(query.split()[:8])
                    self._log(f"  Query truncated to 8 words: {query!r}")
                # Dedup: skip if exact query already searched
                if query.lower() in [s.query.lower() for s in searches]:
                    self._log(f"  Duplicate query {query!r} — skipping")
                    continue
                self._log(
                    f"  Search {i+1}/{MAX_SEARCHES}: '{query}' — {reason}"
                )

            # Execute Tavily search
            try:
                raw = tavily.search(
                    query        = query,
                    search_depth = SEARCH_DEPTH,
                    max_results  = RESULTS_PER_Q,
                )
                results  = raw.get("results", [])
                answer   = raw.get("answer") or ""
                snippets = [
                    {
                        "title":   r.get("title", ""),
                        "url":     r.get("url", ""),
                        "content": r.get("content", "")[:800],  # cap per result
                        "score":   r.get("score", 0.0),
                    }
                    for r in results
                    if r.get("content")
                ]
                searches.append(SearchResult(
                    query    = query,
                    reason   = reason,
                    snippets = snippets,
                    answer   = answer,
                ))
                self._log(
                    f"    → {len(snippets)} results "
                    f"(top score: {snippets[0]['score']:.2f})" if snippets
                    else "    → 0 results"
                )

                # Extract character names from this batch
                characters = self._extract_characters(searches)

            except Exception as e:
                self._log(f"    Tavily error on search {i+1}: {e}")
                continue

            time.sleep(0.3)  # gentle rate limiting

        self._log(
            f"Research complete: {len(searches)} searches, "
            f"{len(characters)} figures identified: {characters}"
        )
        return searches

    def _decide_next_query(
        self,
        topic:      str,
        searches:   list[SearchResult],
        characters: list[str],
        n_done:     int,
    ) -> dict:
        """Ask the LLM what to search for next."""
        # Balanced summary — 2 lines per search, front-capped so early context
        # is preserved. Tail-slicing caused LLM to lose early search context.
        summary_lines = []
        for s in searches:
            summary_lines.append(f"Q: {s.query}")
            if s.snippets:
                best = s.snippets[0]
                summary_lines.append(f"   → {best['title']}: {best['content'][:150]}")
        summary = "\n".join(summary_lines)[:3000]  # front-cap, not tail-slice

        # Identify gaps (simple heuristic: what important aspects are thin)
        gaps = self._identify_gaps(searches, characters)

        user = _NEXT_QUERY_USER.format(
            topic        = topic,
            n_done       = n_done,
            max_searches = MAX_SEARCHES,
            summary      = summary,
            characters   = ", ".join(characters) if characters else "none yet",
            gaps         = gaps,
        )

        try:
            result = self.llm.complete_json(
                _NEXT_QUERY_SYSTEM, user,
                max_tokens       = 512,   # was 256 — caused truncated query strings
                temperature      = 0.7,
                primary_provider = "gemini",
            )
            return result
        except Exception as e:
            self._log(f"    next-query LLM failed: {e}")
            return {"done": True}

    def _identify_gaps(
        self,
        searches:   list[SearchResult],
        characters: list[str],
    ) -> str:
        """Simple heuristic gap analysis — no LLM needed."""
        all_text = " ".join(
            s.answer + " " + " ".join(r["content"] for r in s.snippets)
            for s in searches
        ).lower()

        gaps = []
        if len(characters) < 2:
            gaps.append("Need more named figures / historical actors")
        if "date" not in all_text and "year" not in all_text:
            gaps.append("Need specific dates and timeline")
        if not any(w in all_text for w in ["proof", "equation", "formula",
                                            "theorem", "postulate", "theory"]):
            gaps.append("Need technical/scientific substance")
        if not any(w in all_text for w in ["controversial", "rejected",
                                            "disagreed", "debate", "criticism"]):
            gaps.append("Need controversy/reaction material")
        if len(searches) < 5:
            gaps.append("Still early — more depth needed across all areas")
        return "; ".join(gaps) if gaps else "Story feels fairly complete"

    def _extract_characters(self, searches: list[SearchResult]) -> list[str]:
        """Extract named historical figures mentioned in search results."""
        import re
        all_text = " ".join(
            s.answer + " " + " ".join(r["content"] for r in s.snippets)
            for s in searches
        )
        # Simple: capitalised proper-name patterns
        # Catches "Einstein", "Niels Bohr", "Lobachevsky" etc.
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b', all_text)
        # Filter common false positives
        skip = {"The", "This", "That", "He", "She", "They", "His", "Her",
                "In", "At", "By", "On", "With", "From", "For", "Of",
                "As", "But", "And", "Or", "If", "It", "We", "You",
                "However", "Therefore", "Although", "Since", "While"}
        freq: dict[str, int] = {}
        for n in names:
            if n not in skip and len(n) > 3:
                freq[n] = freq.get(n, 0) + 1
        # Return names mentioned ≥2 times, sorted by frequency
        return [n for n, c in sorted(freq.items(), key=lambda x: -x[1])
                if c >= 2][:8]

    # ── Stage 3: Write the documentary ───────────────────────────────────────

    def _write_documentary(
        self,
        topic:          str,
        searches:       list[SearchResult],
        target_minutes: float,
        presentation_style: str = "tiktok-thriller"
    ) -> DocumentaryScript:
        """LLM composes the full tagged documentary script."""
        self._log("Writing documentary script...")

        # Compile research digest
        digest_parts = []
        for i, s in enumerate(searches, 1):
            digest_parts.append(f"\n─── Search {i}: {s.query} ───")
            if s.answer:
                digest_parts.append(f"Summary: {s.answer}")
            for r in s.snippets:
                digest_parts.append(
                    f"[{r['title']}] (score={r['score']:.2f})\n{r['content']}"
                )
        research_digest = "\n".join(digest_parts)[:12000]  # cap at ~12k

        characters = self._extract_characters(searches)

        # Target paragraph count: ~6 paragraphs per minute (narrator-heavy)
        target_paragraphs = int(target_minutes * 6)
        ############################################
        # ── Load style pack and inject narration voice guidance ───────────
        # ScriptWriter normally applies the style pack but is bypassed for
        # documentary mode. Inject vocal_palette + narration_tone here so
        # Gemini writes with [pauses], Derek Muller voice persona, and
        # forbidden pattern rules baked into the narration text.
        style_guidance = ""
        try:
            import yaml as _yaml, os as _os
            skills_dir = _os.path.join(_os.path.dirname(__file__),
                                       "..", "skills", "styles")
            style_file = _os.path.join(skills_dir,
                                       f"{presentation_style}.yaml")
            if _os.path.exists(style_file):
                with open(style_file, encoding="utf-8") as f:
                    style_pack = _yaml.safe_load(f) or {}

                vocal = style_pack.get("vocal_palette", "")
                tone = style_pack.get("narration_tone", "")

                # Extract just the FORBIDDEN PATTERNS section
                forbidden_section = ""
                if "FORBIDDEN PATTERNS" in (tone or ""):
                    start = tone.find("FORBIDDEN PATTERNS")
                    forbidden_section = tone[start:start + 600].strip()

                if vocal or tone:
                    style_guidance = f"""

        NARRATION VOICE AND TTS STYLE — CRITICAL:
        The following rules govern how the script is WRITTEN for text-to-speech.
        Gemini TTS reads this verbatim. The vocal markers below are real TTS cues.

        {("VOCAL PALETTE:\n" + vocal[:800]) if vocal else ""}

        {("NARRATION PERSONA:\n" + tone[:600]) if tone else ""}

        {("FORBIDDEN (these words signal AI slop — never use them):\n" + forbidden_section) if forbidden_section else ""}

        TTS-SPECIFIC WRITING RULES (non-negotiable):
          • Embed [pauses] before every date, name, ironic reversal, or key fact.
            Example: "In 1935, [pauses] Schrödinger invented a thought experiment."
          • Use sentence fragments for impact: "Blue. [pauses] Nobody could make blue."
          • Max 2 vocal sounds per paragraph. Never open a paragraph with one.
          • Vary sentence length: long context → short impact → long context → short irony.
          • The shortest sentence in each paragraph carries the most emotional weight.
          • Write as if Derek Muller is speaking to one person who almost clicked away.
        """
                self._log(
                    f"Style pack '{presentation_style}' injected into writer prompt "
                    f"({len(style_guidance)} chars)"
                )
            else:
                self._log(
                    f"Style pack '{presentation_style}' not found — "
                    f"using default documentary voice"
                )
        except Exception as e:
            self._log(f"Style pack load failed (non-critical): {e}")
            style_guidance = ""
        ##################################################

        system = _WRITER_SYSTEM.format(
            target_minutes    = target_minutes,
            target_paragraphs = target_paragraphs,
        ) + style_guidance

        user = _WRITER_USER.format(
            topic             = topic,
            target_minutes    = target_minutes,
            target_paragraphs = target_paragraphs,
            n_searches        = len(searches),
            research_digest   = research_digest,
            characters        = ", ".join(characters) if characters else "none",
        )

        result = self.llm.complete_json(
            system, user,
            max_tokens       = 16000,   # 6-8min documentary = ~5-7k words + JSON wrapper
                                           # 8192 truncates the later paragraphs silently.
                                           # Claude Sonnet supports up to 64k output tokens.
            temperature      = 0.85,
            primary_provider = "claude",   # Claude for prose quality
        )

        # Parse paragraphs
        paragraphs = []
        raw_paras  = result.get("paragraphs", [])

        for p in raw_paras:
            ptype   = p.get("type", "NARRATOR").upper()
            speaker = p.get("speaker") or None
            text    = p.get("text", "").strip()

            if not text:
                continue

            # Normalise type
            if ptype not in (P_NARRATOR, P_STORY, P_TECHNICAL, P_VOICE):
                ptype = P_NARRATOR

            paragraphs.append(Paragraph(
                type    = ptype,
                speaker = speaker,
                text    = text,
            ))

        # Validate narrator flow — fix any missing bridges
        paragraphs = self._enforce_narrator_flow(paragraphs)

        script = DocumentaryScript(
            title             = result.get("title", topic),
            tagline           = result.get("tagline", ""),
            topic             = topic,
            suggested_minutes = float(result.get("suggested_minutes",
                                                  target_minutes)),
            suggested_style   = result.get("suggested_style",
                                           "tiktok-thriller"),
            characters        = result.get("characters", characters),
            paragraphs        = paragraphs,
            searches_used     = len(searches),
        )

        self._log(
            f"Script written: {len(paragraphs)} paragraphs, "
            f"suggested style={script.suggested_style}, "
            f"suggested minutes={script.suggested_minutes}"
        )
        return script

    def _enforce_narrator_flow(
        self,
        paragraphs: list[Paragraph],
    ) -> list[Paragraph]:
        """
        Guarantee no two non-NARRATOR paragraphs are adjacent.
        If STORY→TECHNICAL or TECHNICAL→VOICE appears, insert a short
        bridging NARRATOR paragraph between them.
        This fixes LLM compliance failures without a full re-generation.
        """
        if not paragraphs:
            return paragraphs

        fixed  = [paragraphs[0]]
        bridge = Paragraph(
            type    = P_NARRATOR,
            speaker = None,
            text    = "[Narrator bridge — expand in editing]",
        )

        for para in paragraphs[1:]:
            prev = fixed[-1]
            if prev.type != P_NARRATOR and para.type != P_NARRATOR:
                # Insert synthetic bridge
                fixed.append(bridge)
            fixed.append(para)

        # Ensure document ends on a NARRATOR
        if fixed[-1].type != P_NARRATOR:
            fixed.append(Paragraph(
                type    = P_NARRATOR,
                speaker = None,
                text    = (
                    "And so the debate continues — not as a failure of science, "
                    "but as proof that the greatest questions are never truly closed."
                ),
            ))

        return fixed

    # ── Stage 4: PDF assembly ─────────────────────────────────────────────────

    def _build_pdf(self, script: DocumentaryScript, topic: str) -> str:
        """Render the documentary script to a formatted PDF."""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import mm
            from reportlab.lib.colors import HexColor
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph as RLParagraph,
                Spacer, HRFlowable, PageBreak,
            )
            from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
        except ImportError:
            raise RuntimeError(
                "reportlab not installed. Run: pip install reportlab"
            )

        papers_dir = os.path.join(self.cfg.output_root, "papers")
        os.makedirs(papers_dir, exist_ok=True)

        slug     = slugify(topic)[:60]
        filename = f"{slug}_documentary.pdf"
        pdf_path = os.path.join(papers_dir, filename)

        # ── Colour palette ────────────────────────────────────────────────────
        C_BG        = HexColor("#0A0A0F")
        C_WHITE     = HexColor("#F0F0F0")
        C_CYAN      = HexColor("#00D4FF")
        C_PURPLE    = HexColor("#7B2FFF")
        C_GOLD      = HexColor("#FFD700")
        C_GREY      = HexColor("#A0A0B0")
        C_NARRATOR  = HexColor("#00D4FF")   # cyan label
        C_STORY     = HexColor("#F0F0F0")   # white text
        C_TECHNICAL = HexColor("#7BFF6A")   # green tint for tech
        C_VOICE     = HexColor("#FFB347")   # amber for voice

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize       = A4,
            leftMargin     = 20 * mm,
            rightMargin    = 20 * mm,
            topMargin      = 20 * mm,
            bottomMargin   = 20 * mm,
        )

        styles = getSampleStyleSheet()
        base_font = "Helvetica"

        # ── Custom styles ─────────────────────────────────────────────────────
        def S(name, **kw) -> ParagraphStyle:
            return ParagraphStyle(name, **kw)

        sTitle = S("DocTitle",
            fontName    = f"{base_font}-Bold",
            fontSize    = 28,
            textColor   = C_CYAN,
            alignment   = TA_CENTER,
            spaceAfter  = 4 * mm,
        )
        sTagline = S("DocTagline",
            fontName    = f"{base_font}-Oblique",
            fontSize    = 13,
            textColor   = C_GOLD,
            alignment   = TA_CENTER,
            spaceAfter  = 8 * mm,
        )
        sMeta = S("DocMeta",
            fontName    = base_font,
            fontSize    = 9,
            textColor   = C_GREY,
            alignment   = TA_CENTER,
            spaceAfter  = 6 * mm,
        )
        sLabel = S("ParaLabel",
            fontName    = f"{base_font}-Bold",
            fontSize    = 8,
            textColor   = C_GREY,
            spaceAfter  = 1 * mm,
            spaceBefore = 5 * mm,
        )
        sNarrator = S("Narrator",
            fontName    = f"{base_font}-Oblique",
            fontSize    = 11,
            textColor   = C_NARRATOR,
            leading     = 18,
            alignment   = TA_LEFT,
            spaceAfter  = 4 * mm,
            leftIndent  = 8 * mm,
        )
        sStory = S("Story",
            fontName    = base_font,
            fontSize    = 11,
            textColor   = C_STORY,
            leading     = 18,
            alignment   = TA_JUSTIFY,
            spaceAfter  = 4 * mm,
        )
        sTechnical = S("Technical",
            fontName    = f"{base_font}-Bold",
            fontSize    = 11,
            textColor   = C_TECHNICAL,
            leading     = 18,
            alignment   = TA_JUSTIFY,
            spaceAfter  = 4 * mm,
        )
        sVoice = S("Voice",
            fontName    = f"{base_font}-Oblique",
            fontSize    = 12,
            textColor   = C_VOICE,
            leading     = 20,
            alignment   = TA_LEFT,
            spaceAfter  = 5 * mm,
            leftIndent  = 12 * mm,
            rightIndent = 12 * mm,
        )

        # ── Type → (label text, text style) ──────────────────────────────────
        def _para_style(para: Paragraph):
            if para.type == P_NARRATOR:
                label = "▸ NARRATOR"
                style = sNarrator
            elif para.type == P_STORY:
                label = "▸ STORY"
                style = sStory
            elif para.type == P_TECHNICAL:
                label = "▸ TECHNICAL"
                style = sTechnical
            elif para.type == P_VOICE:
                name  = para.speaker or "Unknown"
                label = f"▸ VOICE: {name.upper()}"
                style = sVoice
            else:
                label = "▸ NARRATOR"
                style = sNarrator
            return label, style

        # ── Build story ───────────────────────────────────────────────────────
        story = []

        # Cover
        story.append(Spacer(1, 10 * mm))
        story.append(RLParagraph(script.title, sTitle))
        if script.tagline:
            story.append(RLParagraph(f'"{script.tagline}"', sTagline))
        story.append(HRFlowable(
            width="100%", thickness=1, color=C_PURPLE, spaceAfter=4 * mm
        ))
        story.append(RLParagraph(
            f"Generated {datetime.now().strftime('%B %d, %Y')} · "
            f"{len(script.paragraphs)} paragraphs · "
            f"{script.searches_used} web sources · "
            f"Suggested: {script.suggested_style} · "
            f"{script.suggested_minutes:.0f} min",
            sMeta,
        ))
        if script.characters:
            story.append(RLParagraph(
                "Featuring: " + ", ".join(script.characters),
                sMeta,
            ))
        story.append(Spacer(1, 8 * mm))

        # Paragraphs
        prev_type = None
        for para in script.paragraphs:
            label, pstyle = _para_style(para)

            # Divider before non-narrator after narrator
            if prev_type == P_NARRATOR and para.type != P_NARRATOR:
                story.append(HRFlowable(
                    width="60%", thickness=0.5,
                    color=C_PURPLE, spaceAfter=2 * mm,
                ))

            story.append(RLParagraph(label, sLabel))

            # Escape special ReportLab chars
            safe_text = (
                para.text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            story.append(RLParagraph(safe_text, pstyle))
            prev_type = para.type

        # ── Render ────────────────────────────────────────────────────────────
        doc.build(story)
        self._log(f"PDF written: {pdf_path}")
        return pdf_path

    def run(self, state):
        """Not used in direct pipeline — called via research()."""
        return state