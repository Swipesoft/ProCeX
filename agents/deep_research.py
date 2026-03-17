"""
agents/deep_research.py
Speed-performance Pareto-frontier deep research agent.

Converts a bare topic string into a fully-composed academic PDF report
and saves it to the papers/ directory so the rest of the ProcEx pipeline
can consume it as --input.

Usage (called automatically by main.py when --input is omitted):
    python main.py --topic "Hyperkalemia in the NCLEX" --minutes 5

Internal pipeline (all in this file):
    Stage 1 — Planner      : LLM produces a structured JSON research plan
                             (sections, word targets, image prompts, tables)
    Stage 2 — Section Writer: One focused LLM call per section → full prose
    Stage 3 — Image Gen    : NanoBanana call for planned figures (Gemini only)
    Stage 4 — PDF Assembler: ReportLab composes the final academic PDF

Section count is derived from target_minutes:
    ≤ 3 min  → 4  sections  (fast)
    ≤ 6 min  → 7  sections  (standard)
    ≤ 10 min → 11 sections  (deep)
    > 10 min → 14 sections  (extended)

Provider routing:
    Planner        → gemini (primary) → openai fallback
    Section writer → gemini (primary) → openai fallback
    Image gen      → gemini imagen (NanoBanana) only; graceful skip if unavailable
    Claude is intentionally excluded (may be unavailable; keep Anthropic costs separate)
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from agents.base_agent import BaseAgent
from config import ProcExConfig
from state import ProcExState
from utils.llm_client import LLMClient
from utils.slug import slugify


# ── Section count by target duration ─────────────────────────────────────────

def _section_count(minutes: float) -> int:
    if minutes <= 3:  return 4
    if minutes <= 6:  return 7
    if minutes <= 10: return 11
    return 14


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class TablePlan:
    caption:  str
    columns:  list[str]
    row_hint: str   # brief instruction for what rows should contain
    rows:     list[list[str]] = field(default_factory=list)   # filled in Stage 2


@dataclass
class ImagePlan:
    needed:  bool
    prompt:  str   # NanoBanana prompt
    caption: str
    path:    Optional[str] = None   # filled in Stage 3


@dataclass
class SectionPlan:
    index:      int
    heading:    str
    content_brief: str
    word_target:   int
    image:      ImagePlan
    tables:     list[TablePlan]
    prose:      str = ""   # filled in Stage 2


@dataclass
class ResearchPlan:
    title:    str
    abstract: str
    sections: list[SectionPlan]


# ── Prompts ───────────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """You are an expert academic research planner.
Given a topic, produce a structured JSON research plan for a comprehensive
educational report. Write from your own knowledge — do not reference external
sources or suggest web searches.

CRITICAL: Return ONLY valid JSON. No markdown fences, no preamble.

Schema:
{
  "title": "Full academic title",
  "abstract": "150-word abstract summarising the report",
  "sections": [
    {
      "index": 1,
      "heading": "Section heading",
      "content_brief": "2-3 sentence brief of what this section covers",
      "word_target": 350,
      "image": {
        "needed": true,
        "prompt": "Detailed NanoBanana image generation prompt. Medical illustration style. 16:9. 2K.",
        "caption": "Figure N: Caption text"
      },
      "tables": [
        {
          "caption": "Table N: Caption",
          "columns": ["Column A", "Column B", "Column C"],
          "row_hint": "Brief instruction: what each row should represent"
        }
      ]
    }
  ]
}

Rules:
- image.needed should be true only when a diagram, illustration, or visual
  genuinely aids understanding (anatomy, physiology, clinical algorithms).
  Set to false for purely conceptual or list-based sections.
- tables list may be empty [] if no table is needed for a section.
- word_target: introduction/conclusion = 200-250, body sections = 350-450.
- Do not include a References section — this is an AI knowledge report.
- The abstract is a top-level field, not a section.
"""


_SECTION_WRITER_SYSTEM = """You are an expert educational writer producing
sections of an academic report. Write in clear, precise, professor-voice
English. Use correct clinical/technical terminology. Do not add markdown
headers or bullet points — write flowing prose paragraphs only.

If the section brief mentions a table, write the table rows as a JSON block
at the very end of your response, separated by the delimiter:
  ===TABLES===
followed by a JSON array of tables, each with:
  {"caption": "...", "columns": [...], "rows": [[...], [...]]}

If no tables are needed, write only prose — no delimiter, no JSON.

Do not include the section heading in your response — only the body content.
Write from your own knowledge. Do not reference "this report" or "as stated above".
"""


# ── Main agent class ──────────────────────────────────────────────────────────

class DeepResearchAgent(BaseAgent):
    """
    Deep-research agent. Extends BaseAgent for consistent logging and config
    access. Called via research() not run() — run() is a guard.

    Converts a bare topic string into a fully-composed academic PDF report
    saved to papers/ so the rest of ProcEx can consume it as --input.
    """
    name = "DeepResearch"

    def run(self, state: ProcExState) -> ProcExState:
        raise NotImplementedError(
            "DeepResearchAgent is called via research(topic, target_minutes), not run(state)."
        )

    def __init__(self, cfg: ProcExConfig, llm: LLMClient):
        super().__init__(cfg, llm)
        self._genai = None
        self._init_genai()

    def _init_genai(self):
        if not self.cfg.gemini_api_key:
            self._log("WARNING: GEMINI_API_KEY not set — figures will be skipped")
            return
        try:
            from google import genai
            self._genai = genai.Client(api_key=self.cfg.gemini_api_key)
            self._log("Gemini image client initialised")
        except ImportError:
            self._log("WARNING: google-genai SDK not installed — figures will be skipped")

    # ── Public entry point ────────────────────────────────────────────────────

    def research(self, topic: str, target_minutes: float = 5.0) -> str:
        """
        Run the full research pipeline for the given topic.
        Returns the path to the output PDF in papers/.
        """
        import time as _time
        t0         = _time.time()
        slug       = slugify(topic)
        n_sections = _section_count(target_minutes)

        self._log(f"{'='*55}")
        self._log(f"Topic:      '{topic}'")
        self._log(f"Sections:   {n_sections}  |  Target: {target_minutes} min  |  Slug: {slug}")
        self._log(f"{'='*55}")

        # ── Stage 1: Plan ─────────────────────────────────────────────────────
        self._log("Stage 1/4 ▶ Planning research structure...")
        plan = self._plan(topic, n_sections)
        n_figs_planned  = sum(1 for s in plan.sections if s.image.needed)
        n_tables_planned = sum(len(s.tables) for s in plan.sections)
        self._log(f"Stage 1/4 ✓ Plan ready")
        self._log(f"  Title:    {plan.title}")
        self._log(f"  Sections: {len(plan.sections)}  |  Figures planned: {n_figs_planned}"
                  f"  |  Tables planned: {n_tables_planned}")
        for s in plan.sections:
            img_tag   = "📷" if s.image.needed else "  "
            tbl_tag   = f"[{len(s.tables)}T]" if s.tables else "    "
            self._log(f"    {img_tag} {tbl_tag}  §{s.index:02d} {s.heading}  (~{s.word_target}w)")

        # ── Stage 2: Write sections ───────────────────────────────────────────
        self._log("Stage 2/4 ▶ Writing sections (3 workers)...")
        self._write_sections(plan)
        total_words = sum(len(s.prose.split()) for s in plan.sections)
        self._log(f"Stage 2/4 ✓ Writing complete — {total_words} words total")

        # ── Stage 3: Generate images ──────────────────────────────────────────
        self._log(f"Stage 3/4 ▶ Generating {n_figs_planned} figure(s) via NanoBanana...")
        img_dir = os.path.join(self.cfg.output_root, "research_images", slug)
        os.makedirs(img_dir, exist_ok=True)
        self._generate_images(plan, img_dir)
        n_figs_done = sum(1 for s in plan.sections if s.image.needed and s.image.path)
        self._log(f"Stage 3/4 ✓ Figures: {n_figs_done}/{n_figs_planned} generated")

        # ── Stage 4: Assemble PDF ─────────────────────────────────────────────
        self._log("Stage 4/4 ▶ Assembling PDF with ReportLab...")
        papers_dir = "papers"
        os.makedirs(papers_dir, exist_ok=True)
        pdf_path = os.path.join(papers_dir, f"{slug}.pdf")
        self._assemble_pdf(plan, pdf_path, topic)
        size_kb  = os.path.getsize(pdf_path) // 1024
        elapsed  = _time.time() - t0
        self._log(f"Stage 4/4 ✓ PDF assembled")
        self._log(f"{'='*55}")
        self._log(f"Done in {elapsed:.1f}s  |  {size_kb} KB  |  {pdf_path}")
        self._log(f"{'='*55}")

        return pdf_path

    # ── Stage 1: Planner ─────────────────────────────────────────────────────

    def _plan(self, topic: str, n_sections: int) -> ResearchPlan:
        user_prompt = (
            f"Topic: {topic}\n"
            f"Required section count (excluding abstract): {n_sections}\n"
            f"Produce the complete research plan JSON now."
        )

        for attempt in range(self.cfg.max_llm_retries):
            self._log(f"  Planner attempt {attempt+1}/{self.cfg.max_llm_retries}...")
            try:
                data = self.llm.complete_json(
                    _PLANNER_SYSTEM,
                    user_prompt,
                    max_tokens=8192,
                    temperature=0.4,
                    primary_provider="gemini",
                )
                self._log(f"  Planner LLM call succeeded")
                break
            except Exception as e:
                self._log(f"  Planner attempt {attempt+1} failed: {e}")
                if attempt == self.cfg.max_llm_retries - 1:
                    raise RuntimeError(f"Planner failed after {self.cfg.max_llm_retries} attempts") from e
                time.sleep(2)

        return self._parse_plan(data)

    def _parse_plan(self, data: dict) -> ResearchPlan:
        sections = []
        for raw in data.get("sections", []):
            img_raw = raw.get("image", {})
            image   = ImagePlan(
                needed  = bool(img_raw.get("needed", False)),
                prompt  = str(img_raw.get("prompt", "")),
                caption = str(img_raw.get("caption", "")),
            )
            tables = [
                TablePlan(
                    caption  = t.get("caption", ""),
                    columns  = t.get("columns", []),
                    row_hint = t.get("row_hint", ""),
                )
                for t in raw.get("tables", [])
                if isinstance(t, dict)
            ]
            sections.append(SectionPlan(
                index         = int(raw.get("index", len(sections) + 1)),
                heading       = str(raw.get("heading", f"Section {len(sections)+1}")),
                content_brief = str(raw.get("content_brief", "")),
                word_target   = int(raw.get("word_target", 350)),
                image         = image,
                tables        = tables,
            ))

        return ResearchPlan(
            title    = str(data.get("title", "Research Report")),
            abstract = str(data.get("abstract", "")),
            sections = sections,
        )

    # ── Stage 2: Section writer ───────────────────────────────────────────────

    def _write_sections(self, plan: ResearchPlan) -> None:
        def write_one(section: SectionPlan) -> SectionPlan:
            self._log(f"  §{section.index:02d} '{section.heading}' — writing (~{section.word_target}w)...")
            table_instruction = ""
            if section.tables:
                descs = "; ".join(
                    f"{t.caption} (columns: {', '.join(t.columns)}) — {t.row_hint}"
                    for t in section.tables
                )
                table_instruction = (
                    f"\n\nAfter the prose, include the following table(s) as a "
                    f"JSON block after the ===TABLES=== delimiter:\n{descs}"
                )

            user_prompt = (
                f"Report title: {plan.title}\n"
                f"Section heading: {section.heading}\n"
                f"Content brief: {section.content_brief}\n"
                f"Target word count: {section.word_target} words\n"
                f"{table_instruction}\n\n"
                f"Write the section now."
            )

            for attempt in range(self.cfg.max_llm_retries):
                try:
                    raw = self.llm.complete(
                        _SECTION_WRITER_SYSTEM,
                        user_prompt,
                        max_tokens=4096,
                        temperature=0.6,
                        primary_provider="gemini",
                    )
                    break
                except Exception as e:
                    self._log(f"  §{section.index:02d} attempt {attempt+1} failed: {e}")
                    if attempt == self.cfg.max_llm_retries - 1:
                        raw = f"{section.heading}\n\n{section.content_brief}"
                    time.sleep(1)

            # Split prose from tables
            if "===TABLES===" in raw:
                prose_part, tables_part = raw.split("===TABLES===", 1)
                section.prose = prose_part.strip()
                self._parse_table_rows(section, tables_part.strip())
                n_rows = sum(len(t.rows) for t in section.tables)
                self._log(f"  §{section.index:02d} ✓  {len(section.prose.split())}w prose "
                          f"+ {len(section.tables)} table(s) / {n_rows} rows")
            else:
                section.prose = raw.strip()
                self._log(f"  §{section.index:02d} ✓  {len(section.prose.split())}w prose")

            return section

        # Run up to 3 section writers concurrently
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(write_one, s): s for s in plan.sections}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    s = futures[future]
                    self._log(f"Section {s.index} write failed: {e}")
                    s.prose = s.content_brief   # graceful fallback

    def _parse_table_rows(self, section: SectionPlan, tables_json: str) -> None:
        """Parse the ===TABLES=== JSON block and fill TablePlan.rows."""
        try:
            tables_json = re.sub(r"^```(?:json)?\s*", "", tables_json.strip(), flags=re.MULTILINE)
            tables_json = re.sub(r"```\s*$",           "", tables_json.strip(), flags=re.MULTILINE)
            parsed = json.loads(tables_json.strip())
            if isinstance(parsed, dict):
                parsed = [parsed]
            for i, t_data in enumerate(parsed):
                if i < len(section.tables):
                    section.tables[i].rows = [
                        [str(cell) for cell in row]
                        for row in t_data.get("rows", [])
                    ]
        except Exception as e:
            self._log(f"  Section {section.index}: table JSON parse failed ({e}) — skipping tables")

    # ── Stage 3: Image generation ─────────────────────────────────────────────

    def _generate_images(self, plan: ResearchPlan, img_dir: str) -> None:
        if self._genai is None:
            self._log("  No Gemini client — skipping all figures")
            return

        from google.genai import types

        targets = [s for s in plan.sections if s.image.needed and s.image.prompt]
        if not targets:
            self._log("  No figures requested by plan")
            return

        for section in targets:
            self._log(f"  §{section.index:02d} figure: calling NanoBanana...")
            img_path = os.path.join(img_dir, f"section_{section.index:02d}.png")
            prompt   = self._enrich_image_prompt(section.image.prompt)

            try:
                response = self._genai.models.generate_content(
                    model    = self.cfg.nano_pro_model,
                    contents = prompt,
                    config   = types.GenerateContentConfig(
                        response_modalities=["image", "text"],
                    ),
                )

                saved = False
                for part in response.parts:
                    if hasattr(part, "inline_data") and part.inline_data is not None:
                        img_bytes = part.inline_data.data
                        if isinstance(img_bytes, str):
                            img_bytes = base64.b64decode(img_bytes)
                        with open(img_path, "wb") as f:
                            f.write(img_bytes)
                        saved = True
                        break

                if saved:
                    size_kb = os.path.getsize(img_path) // 1024
                    section.image.path = img_path
                    self._log(f"  §{section.index:02d} figure ✓  {size_kb} KB → {img_path}")
                else:
                    self._log(f"  §{section.index:02d} figure ✗  NanoBanana returned no image data")

            except Exception as e:
                self._log(f"  §{section.index:02d} figure ✗  {e}")

    @staticmethod
    def _enrich_image_prompt(prompt: str, aspect: str = "16:9") -> str:
        additions = []
        if aspect not in prompt:
            additions.append(f"{aspect} aspect ratio")
        if "2K" not in prompt and "4K" not in prompt:
            additions.append("2K resolution")
        if "medical illustration" not in prompt.lower() and "illustration" not in prompt.lower():
            additions.append("clean educational illustration style")
        if additions:
            prompt = prompt.rstrip(". ") + ". " + ". ".join(additions) + "."
        return prompt

    # ── Stage 4: PDF assembly ─────────────────────────────────────────────────

    def _assemble_pdf(self, plan: ResearchPlan, output_path: str, topic: str) -> None:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, PageBreak,
            Table, TableStyle, Image as RLImage, HRFlowable,
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

        # ── Page layout ───────────────────────────────────────────────────────
        doc = SimpleDocTemplate(
            output_path,
            pagesize      = A4,
            leftMargin    = 2.5 * cm,
            rightMargin   = 2.5 * cm,
            topMargin     = 2.5 * cm,
            bottomMargin  = 2.5 * cm,
        )

        W, _ = A4
        usable_w = W - 5 * cm

        # ── Style sheet ───────────────────────────────────────────────────────
        base = getSampleStyleSheet()

        def _style(name, parent="Normal", **kwargs) -> ParagraphStyle:
            return ParagraphStyle(name, parent=base[parent], **kwargs)

        S = {
            "cover_title": _style("cover_title", "Title",
                fontSize=28, leading=36, textColor=colors.HexColor("#0A1628"),
                alignment=TA_CENTER, spaceAfter=12),
            "cover_sub": _style("cover_sub", "Normal",
                fontSize=13, textColor=colors.HexColor("#4A5568"),
                alignment=TA_CENTER, spaceAfter=6),
            "cover_date": _style("cover_date", "Normal",
                fontSize=10, textColor=colors.HexColor("#718096"),
                alignment=TA_CENTER),
            "abstract_heading": _style("abstract_heading", "Heading1",
                fontSize=13, textColor=colors.HexColor("#2D3748"),
                spaceAfter=8, spaceBefore=20),
            "abstract_body": _style("abstract_body", "Normal",
                fontSize=10.5, leading=16, textColor=colors.HexColor("#2D3748"),
                alignment=TA_JUSTIFY),
            "section_heading": _style("section_heading", "Heading1",
                fontSize=15, leading=20, textColor=colors.HexColor("#1A365D"),
                spaceBefore=22, spaceAfter=8,
                borderPad=4),
            "body": _style("body", "Normal",
                fontSize=10.5, leading=17, textColor=colors.HexColor("#2D3748"),
                alignment=TA_JUSTIFY, spaceAfter=10),
            "caption": _style("caption", "Normal",
                fontSize=9, leading=13, textColor=colors.HexColor("#718096"),
                alignment=TA_CENTER, spaceBefore=4, spaceAfter=14),
            "table_header": _style("table_header", "Normal",
                fontSize=9.5, textColor=colors.white, fontName="Helvetica-Bold"),
            "table_cell": _style("table_cell", "Normal",
                fontSize=9, leading=13, textColor=colors.HexColor("#2D3748")),
            "footer_note": _style("footer_note", "Normal",
                fontSize=8.5, textColor=colors.HexColor("#A0AEC0"),
                alignment=TA_CENTER, spaceBefore=30),
        }

        story = []

        # ── Cover page ────────────────────────────────────────────────────────
        story.append(Spacer(1, 3 * cm))
        story.append(HRFlowable(
            width="100%", thickness=3,
            color=colors.HexColor("#1A365D"), spaceAfter=30,
        ))
        story.append(Paragraph(_escape(plan.title), S["cover_title"]))
        story.append(Spacer(1, 0.4 * cm))
        story.append(HRFlowable(
            width="60%", thickness=1,
            color=colors.HexColor("#4A90D9"), spaceAfter=20,
        ))
        story.append(Paragraph("ProCeX DeepResearch Report", S["cover_sub"]))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y')}",
            S["cover_date"],
        ))
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(
            f"Sections: {len(plan.sections)}  ·  "
            f"Figures: {sum(1 for s in plan.sections if s.image.path)}  ·  "
            f"Tables: {sum(len(s.tables) for s in plan.sections)}",
            S["cover_date"],
        ))
        story.append(PageBreak())

        # ── Abstract ──────────────────────────────────────────────────────────
        story.append(Paragraph("Abstract", S["abstract_heading"]))
        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=colors.HexColor("#CBD5E0"), spaceAfter=12,
        ))
        story.append(Paragraph(_escape(plan.abstract), S["abstract_body"]))
        story.append(Spacer(1, 0.5 * cm))

        # ── Sections ──────────────────────────────────────────────────────────
        for section in plan.sections:
            story.append(Paragraph(
                f"{section.index}. {_escape(section.heading)}",
                S["section_heading"],
            ))
            story.append(HRFlowable(
                width="100%", thickness=0.4,
                color=colors.HexColor("#E2E8F0"), spaceAfter=10,
            ))

            # Prose paragraphs
            for para in _split_paragraphs(section.prose):
                story.append(Paragraph(_escape(para), S["body"]))

            # Figure (if generated)
            if section.image.needed and section.image.path and \
               os.path.exists(section.image.path):
                try:
                    img = RLImage(section.image.path, width=usable_w, height=usable_w * 9/16)
                    story.append(Spacer(1, 0.3 * cm))
                    story.append(img)
                    if section.image.caption:
                        story.append(Paragraph(
                            _escape(section.image.caption), S["caption"],
                        ))
                except Exception as e:
                    self._log(f"  Section {section.index}: figure embed failed ({e})")

            # Tables
            for tbl in section.tables:
                if not tbl.rows:
                    continue
                story.append(Spacer(1, 0.3 * cm))
                story.append(self._build_table(tbl, S, usable_w))
                story.append(Paragraph(_escape(tbl.caption), S["caption"]))

        # ── Footer note ───────────────────────────────────────────────────────
        story.append(Spacer(1, 1 * cm))
        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=colors.HexColor("#E2E8F0"), spaceAfter=8,
        ))
        story.append(Paragraph(
            "(C)ProCeX DeepResearch 2026",
            S["footer_note"],
        ))

        doc.build(story)

    @staticmethod
    def _build_table(tbl: TablePlan, S: dict, usable_w: float):
        from reportlab.platypus import Table, TableStyle, Paragraph
        from reportlab.lib import colors

        n_cols = len(tbl.columns)
        col_w  = usable_w / n_cols if n_cols else usable_w

        # Header row + data rows
        header = [Paragraph(_escape(c), S["table_header"]) for c in tbl.columns]
        data   = [header] + [
            [Paragraph(_escape(str(cell)), S["table_cell"]) for cell in row]
            for row in tbl.rows
        ]

        table = Table(data, colWidths=[col_w] * n_cols, repeatRows=1)
        table.setStyle(TableStyle([
            # Header
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1A365D")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0), 9.5),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING",    (0, 0), (-1, 0), 8),
            # Body alternating rows
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#F7FAFC"), colors.white]),
            ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",    (0, 1), (-1, -1), 9),
            ("TOPPADDING",  (0, 1), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            # Grid
            ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E0")),
            ("BOX",         (0, 0), (-1, -1), 0.8, colors.HexColor("#A0AEC0")),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return table


# ── Helpers ───────────────────────────────────────────────────────────────────

def _escape(text: str) -> str:
    """Escape HTML special chars for ReportLab Paragraph."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def _split_paragraphs(prose: str) -> list[str]:
    """Split prose into non-empty paragraphs on blank lines."""
    paras = re.split(r"\n{2,}", prose.strip())
    return [p.replace("\n", " ").strip() for p in paras if p.strip()]

