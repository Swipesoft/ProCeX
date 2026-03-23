"""
config.py — API keys, model identifiers, resolution presets, directory layout.
All runtime config lives here. Load once in orchestrator, pass around.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# Resolution Presets
# ─────────────────────────────────────────────

@dataclass
class ResolutionConfig:
    width:          int
    height:         int
    manim_flag:     str    # passed to manim CLI: -ql / -qh / -qk
    ffmpeg_scale:   str    # e.g. "1920:1080"
    nano_res:       str    # NanoBanana resolution hint: "1K" / "2K" / "4K"
    zoompan_frames: int    # FFmpeg zoompan frame count (25fps assumed)

    @property
    def aspect_ratio(self) -> str:
        """Computed from actual dimensions — works for both 16:9 and 9:16."""
        return "9:16" if self.height > self.width else "16:9"

    @property
    def is_portrait(self) -> bool:
        return self.height > self.width

    @property
    def manim_frame_width(self) -> float:
        """Manim canvas width in units. Portrait flips the 14×8 default."""
        return 8.0 if self.is_portrait else 14.0

    @property
    def manim_frame_height(self) -> float:
        """Manim canvas height in units. Portrait flips the 14×8 default."""
        return 14.0 if self.is_portrait else 8.0


RESOLUTIONS: dict[str, ResolutionConfig] = {
    # ── Landscape 16:9 ────────────────────────────────────────────────────────
    "720p": ResolutionConfig(
        width=1280, height=720,
        manim_flag="-ql",
        ffmpeg_scale="1280:720",
        nano_res="1K",
        zoompan_frames=25,
    ),
    "1080p": ResolutionConfig(
        width=1920, height=1080,
        manim_flag="-qh",
        ffmpeg_scale="1920:1080",
        nano_res="2K",
        zoompan_frames=25,
    ),
    "4K": ResolutionConfig(
        width=3840, height=2160,
        manim_flag="-qk",
        ffmpeg_scale="3840:2160",
        nano_res="4K",
        zoompan_frames=25,
    ),
    # ── Portrait 9:16 (vertical / mobile / Reels / Shorts) ───────────────────
    "720p_v": ResolutionConfig(
        width=720, height=1280,
        manim_flag="-ql",
        ffmpeg_scale="720:1280",
        nano_res="1K",
        zoompan_frames=25,
    ),
    "1080p_v": ResolutionConfig(
        width=1080, height=1920,
        manim_flag="-qh",
        ffmpeg_scale="1080:1920",
        nano_res="2K",
        zoompan_frames=25,
    ),
    "4K_v": ResolutionConfig(
        width=2160, height=3840,
        manim_flag="-qk",
        ffmpeg_scale="2160:3840",
        nano_res="4K",
        zoompan_frames=25,
    ),
}


# ─────────────────────────────────────────────
# Cinematic Colour Palette (injected into every Manim prompt)
# ─────────────────────────────────────────────

CINEMATIC_PALETTE = {
    "background":  "#0A0A0F",   # near-black with cool blue undertone
    "text_primary":"#F0F0FF",   # soft white
    "accent_cyan": "#00D4FF",   # electric cyan — primary highlights
    "accent_purple":"#7B2FFF",  # vivid purple — secondary highlights
    "accent_orange":"#FF6B35",  # warm orange — tertiary / warnings
    "accent_green": "#00FF88",  # mint green — success / flow direction
    "accent_gold":  "#FFD700",  # gold — emphasis / NCLEX clinical priority
    "dim_text":     "#8890A4",  # muted blue-grey — secondary labels
    "grid_line":    "#1A1A2E",  # subtle grid
}

MANIM_PALETTE_BLOCK = """
# ── Cinematic Palette (ManimColor objects — required for interpolate_color) ─
BG          = ManimColor("#0A0A0F")
WHITE       = ManimColor("#F0F0FF")
CYAN        = ManimColor("#00D4FF")
PURPLE      = ManimColor("#7B2FFF")
ORANGE      = ManimColor("#FF6B35")
GREEN       = ManimColor("#00FF88")
GOLD        = ManimColor("#FFD700")
DIM         = ManimColor("#8890A4")
# ──────────────────────────────────────────────────────────────────────────
"""


# ─────────────────────────────────────────────
# Main Config
# ─────────────────────────────────────────────

@dataclass
class ProcExConfig:
    # ── API Keys ──────────────────────────────
    anthropic_api_key:   str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    gemini_api_key:      str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    openai_api_key:      str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    elevenlabs_api_key:  str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))

    # ── LLM Models (text) ─────────────────────
    # Fallback chain: Claude → Gemini → OpenAI
    claude_model:        str = "claude-sonnet-4-6"
    gemini_text_model:   str = "gemini-3-flash-preview" #"gemini-3.1-pro-preview"
    openai_model:        str = "gpt-5.4-2026-03-05"

    # ── Per-agent primary LLM ────────────────────────────────────────────────
    # Each agent starts its provider chain with the named primary.
    # The other two providers remain as fallbacks in the default order.
    # Valid values: "claude" | "gemini" | "openai"
    #
    # Routing rationale:
    #   gemini  → ScriptWriter, VisualDirector  (strong at long-form synthesis)
    #   claude  → ManimCoder, DomainRouter       (strong at structured code + reasoning)
    #   claude  → everything else (default)
    agent_primary_llm: dict = None   # populated in __post_init__

    def __post_init__(self):
        if self.agent_primary_llm is None:
            self.agent_primary_llm = {
                "ScriptWriter":   "gemini",   # transcript/narration generation
                "VisualDirector": "gemini",   # scene-level creative direction
                "DomainRouter":   "claude",   # structured classification
                "ManimCoder":     "claude",   # code generation
                "TTSAgent":       "claude",   # minimal LLM use
                "ImageGenAgent":  "gemini",   # image prompt enrichment
                "RendererAgent":  "claude",
                "AssemblerAgent": "claude",
            }
        self.make_dirs()

    # ── Image Generation (NanoBanana) ─────────
    # Pro  → detailed anatomy, labeled diagrams, search grounding
    # Fast → background images, simpler visuals, hybrid scenes
    nano_pro_model:      str = "gemini-3.1-flash-image-preview" #"gemini-3-pro-image-preview"
    nano_fast_model:     str = "gemini-3.1-flash-image-preview"

    # ── ElevenLabs (legacy) ───────────────────
    elevenlabs_voice_id: str = "JBFqnCBsd6RMkjVDRZzb"
    elevenlabs_model:    str = "eleven_multilingual_v2"

    # ── OpenAI TTS ────────────────────────────
    # tts-1    = $15/1M chars (~$0.15 per 10-min video)
    # tts-1-hd = $30/1M chars (~$0.30 per 10-min video, higher quality)
    # Voices: onyx (deep/authoritative), nova, alloy, echo, fable, shimmer
    openai_tts_model:    str = "tts-1"
    openai_tts_voice:    str = "onyx"

    # ── Gemini TTS (default) ──────────────────
    # gemini-2.5-flash-preview-tts — high-quality, expressive narration
    # Voices: Aoede, Charon, Fenrir, Kore, Orus, Puck, Schedar, Sulafat...
    gemini_tts_model:    str = "gemini-2.5-flash-preview-tts"
    gemini_tts_voice:    str = "Aoede"

    # ── TTS provider selection ────────────────
    # "gemini"  → Gemini 2.5 Flash TTS (default, uses GEMINI_API_KEY)
    # "openai"  → OpenAI TTS (fallback, uses OPENAI_API_KEY)
    tts_provider:        str = "gemini"

    # ── Presentation style ────────────────────────────────────────────────────
    # "auto"            → LLM selects based on topic + domain + duration
    # "tiktok-scifi"    → sci-fi dramatic, 45s-1.5min
    # "tiktok-thriller" → historical thriller story, 45s-2min
    # "youtube-tutorial"→ warm deep explainer, 4-10min
    presentation_style: str = "auto"

    # ── Pipeline ──────────────────────────────
    max_llm_retries:     int   = 3

    # ── Parallel worker counts ─────────────────────────────────────────
    tts_workers:         int   = 4   # parallel TTS API calls per scene
    coder_workers:       int   = 4   # parallel ManimCoder LLM calls
    image_workers:       int   = 3   # parallel ImageGen API calls
    render_workers:      int   = 2   # parallel Manim subprocesses
    # render_workers: keep ≤ cpu_count//2 — Manim is CPU-intensive at 4K
    manim_timeout_secs:  int   = 420       # 7 min per scene — generous for complex 4K renders
    enable_critic_loop:  bool  = False
    scenes_per_minute:   float = 2.5       # ~24s avg per scene; tuned for engagement
    subscene_split_threshold_secs: float = 40.0  # scenes longer than this may be split into subscenes

    # ── Output dirs ───────────────────────────
    output_root:    str = "output"

    @property
    def dirs(self) -> dict[str, str]:
        r = self.output_root
        return {
            "root":        r,
            "scenes":      f"{r}/scenes",
            "audio":       f"{r}/audio",
            "images":      f"{r}/images",
            "videos":      f"{r}/videos",
            "manim":       f"{r}/manim",
            "checkpoints": f"{r}/checkpoints",
        }

    def make_dirs(self) -> None:
        import os
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

    def validate(self) -> list[str]:
        """Return list of missing API key warnings."""
        issues = []
        if not self.anthropic_api_key:
            issues.append("ANTHROPIC_API_KEY not set — Claude unavailable")
        if not self.gemini_api_key:
            issues.append("GEMINI_API_KEY not set — Gemini + NanoBanana unavailable")
        if not self.openai_api_key:
            issues.append("OPENAI_API_KEY not set — OpenAI fallback unavailable")
        if not self.elevenlabs_api_key and not self.openai_api_key and not self.gemini_api_key:
            issues.append("No TTS key set — set GEMINI_API_KEY (recommended) or OPENAI_API_KEY")
        return issues
