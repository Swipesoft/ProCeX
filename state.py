"""
ProcEx — Procedural Cinematic Explainer Pipeline
state.py: All shared dataclasses. Everything flows through ProcExState.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class VisualStrategy(str, Enum):
    MANIM            = "MANIM"             # Pure Manim: equations, graphs, ML diagrams, flowcharts
    IMAGE_GEN        = "IMAGE_GEN"         # NanoBanana: anatomical structures, real-world medical imagery
    IMAGE_MANIM_HYBRID = "IMAGE_GEN"  # Retired — maps to IMAGE_GEN; NanoBanana handles labels natively
    TEXT_ANIMATION   = "TEXT_ANIMATION"    # Manim but just cinematic title/quote cards


class Domain(str, Enum):
    ML_MATH          = "ML_MATH"
    MEDICAL          = "MEDICAL"
    CS_SYSTEMS       = "CS_SYSTEMS"
    NCLEX_NURSING    = "NCLEX_NURSING"
    HYBRID           = "HYBRID"


class InputType(str, Enum):
    TEXT = "text"
    PDF  = "pdf"
    URL  = "url"


# ─────────────────────────────────────────────
# Timestamp & Scene
# ─────────────────────────────────────────────

@dataclass
class WordTimestamp:
    word:  str
    start: float   # seconds
    end:   float   # seconds


@dataclass
class Scene:
    id:                 int
    narration_text:     str
    duration_seconds:   float

    # Visual Director decisions
    visual_strategy:    VisualStrategy = VisualStrategy.MANIM
    visual_prompt:      str = ""        # Detailed prompt for NanoBanana / Manim
    visual_reasoning:   str = ""        # Why the director chose this strategy
    needs_labels:       bool = False    # For IMAGE_GEN: add PIL callout labels?
    label_list:         list = field(default_factory=list)  # ["glomerulus", "Bowman's capsule", ...]

    # Manim
    manim_code:         Optional[str] = None
    manim_class_name:   Optional[str] = None
    manim_file_path:    Optional[str] = None

    # Image gen
    image_paths:        list = field(default_factory=list)

    # Timestamps scoped to this scene (subset of global)
    timestamps:         list = field(default_factory=list)  # list[WordTimestamp]

    # TTS audio for this scene (set by TTSAgent)
    tts_audio_path:     str   = ""
    tts_duration:       float = 0.0
    tts_audio_start:    float = 0.0   # start offset within tts_audio_path (for subscene beats)

    # Render output
    clip_path:          Optional[str] = None
    render_attempts:    int = 0
    render_error:       Optional[str] = None

    # VLM Critic fields
    element_count:       int  = 0      # set by VisualDirector; used to gate critic
    _split_recommended:  bool = False  # set by VLMCritic; handled by orchestrator

    # Zone allocation — binding layout contract set by VisualDirector
    # Maps element label → zone name, e.g. {"scene_title": "TITLE", "formula": "MAIN"}
    # ManimCoder treats this as a hard positional contract, not a suggestion.
    zone_allocation:     dict = field(default_factory=dict)

    # Subscene fields — set when VisualDirector splits a long scene into beats
    parent_scene_id:     Optional[int] = None   # original scene id this was split from
    subscene_index:      int = 0                 # position within parent (0 = not a subscene)
    split_depth:         int = 0                 # 0=original, 1=child of split; depth>=1 → ImageGen fallback instead of further split

    # Critic feedback loop — capped at 2 reroutes per scene
    critic_reroute_attempts: int = 0

    # ── Documentary multi-voice fields ───────────────────────────────────────
    # Set by the documentary PDF parser before TTSAgent runs.
    # tts_voice: Gemini voice name override for this scene.
    #   ""        → use cfg.gemini_tts_voice (Aoede) — default for all non-documentary scenes
    #   "Charon"  → male voice for [VOICE: X] critic/character paragraphs
    #   Any valid Gemini prebuilt voice name works.
    tts_voice: str = ""

    # paragraph_type: the documentary structural role of this scene's narration.
    #   ""          → unset, not a documentary scene — VisualDirector uses normal logic
    #   "NARRATOR"  → short bridge narration → TEXT_ANIMATION preferred
    #   "STORY"     → historical/biographical prose → MANIM or IMAGE_GEN (person/place)
    #   "TECHNICAL" → science/math tutoring → MANIM strongly preferred (equations)
    #   "VOICE"     → first-person character voice → IMAGE_GEN strongly preferred (portrait)
    paragraph_type: str = ""


# ─────────────────────────────────────────────
# Pipeline State
# ─────────────────────────────────────────────

@dataclass
class ProcExState:
    # ── Input ──────────────────────────────────
    raw_input:               str = ""
    input_type:              InputType = InputType.TEXT
    topic_slug:              str = ""          # e.g. "attention_mechanism"
    resolution:              str = "1080p"     # "720p" | "1080p" | "4K"
    target_duration_minutes: float = 5.0

    # ── Domain classification ───────────────────
    domain:     Domain = Domain.ML_MATH
    skill_pack:  dict   = field(default_factory=dict)
    style_pack:  dict   = field(default_factory=dict)   # presentation style skill (tiktok-scifi etc.)
    presentation_style: str = "auto"                    # resolved style id

    # ── Script ─────────────────────────────────
    scenes:           list  = field(default_factory=list)   # list[Scene]
    full_script_text: str   = ""

    # ── Audio ───────────────────────────────────
    audio_path:           str   = ""
    all_timestamps:       list  = field(default_factory=list)  # list[WordTimestamp]
    total_audio_duration: float = 0.0

    # ── Rendering ──────────────────────────────
    rendered_clips: list = field(default_factory=list)  # ordered paths

    # ── Output ─────────────────────────────────
    final_video_path: str = ""

    # ── Pipeline health ────────────────────────
    checkpoint_path: str  = ""
    errors:          list = field(default_factory=list)

    # ── Helpers ────────────────────────────────
    def save_checkpoint(self, path: str) -> None:
        """Serialize state to JSON for resume-on-failure."""
        import dataclasses
        def _default(obj):
            if isinstance(obj, Enum):
                return obj.value
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
            return str(obj)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(self), f, default=_default, indent=2)
        self.checkpoint_path = path

    @classmethod
    def load_checkpoint(cls, path: str) -> "ProcExState":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # ── Reconstruct nested dataclasses from plain dicts ───────────────
        # Scenes: list of dicts → list of Scene objects
        raw_scenes = data.pop("scenes", [])
        reconstructed_scenes = []
        for s in raw_scenes:
            if isinstance(s, dict):
                # Reconstruct per-scene WordTimestamps
                s["timestamps"] = [
                    WordTimestamp(**t) if isinstance(t, dict) else t
                    for t in s.get("timestamps", [])
                ]
                # Restore enums
                if isinstance(s.get("visual_strategy"), str):
                    try:
                        s["visual_strategy"] = VisualStrategy(s["visual_strategy"])
                    except ValueError:
                        s["visual_strategy"] = VisualStrategy.MANIM
                # Drop any keys that Scene doesn't know about (forward compat)
                valid_keys = Scene.__dataclass_fields__.keys()
                s = {k: v for k, v in s.items() if k in valid_keys}
                reconstructed_scenes.append(Scene(**s))
            else:
                reconstructed_scenes.append(s)

        # Global timestamps: list of dicts → list of WordTimestamp objects
        raw_ts = data.pop("all_timestamps", [])
        reconstructed_ts = [
            WordTimestamp(**t) if isinstance(t, dict) else t
            for t in raw_ts
        ]

        # Restore top-level enums
        if isinstance(data.get("domain"), str):
            try:
                data["domain"] = Domain(data["domain"])
            except ValueError:
                data["domain"] = Domain.ML_MATH

        if isinstance(data.get("input_type"), str):
            try:
                data["input_type"] = InputType(data["input_type"])
            except ValueError:
                data["input_type"] = InputType.TEXT

        # Build state — only pass fields that ProcExState knows about
        valid_keys = cls.__dataclass_fields__.keys()
        filtered   = {k: v for k, v in data.items() if k in valid_keys}

        state                = cls(**filtered)
        state.scenes         = reconstructed_scenes
        state.all_timestamps = reconstructed_ts
        return state

    def log_error(self, agent: str, msg: str) -> None:
        self.errors.append({"agent": agent, "error": msg})
        print(f"[ERROR] {agent}: {msg}")