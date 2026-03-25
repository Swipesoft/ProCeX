"""
utils/manim_skill_selector.py

Selects and loads relevant manimce-best-practices rule files for a given scene.
Reads .md files from skills/manim_skill/rules/ (cloned from adithya-s-k/manim_skill).

Design:
  - Tier 1 (ALWAYS): small universally-needed rules injected on every call
  - Tier 2 (CONDITIONAL): triggered by keywords in scene.visual_prompt
  - Tier 3 (SKIP): files whose content overlaps with our existing rules
                   (config.md, cli.md, 3d.md, camera.md — we own these better)

The selector is deliberately keyword-based (no LLM call) — fast, deterministic,
and zero additional API cost.

Tier 1 always-on rules (combined ~600 tokens):
  animations.md         — Create vs Write vs FadeIn — most common crash source
  creation-animations.md — DrawBorderThenFill, GrowArrow, correct entrance API
  timing.md             — run_time, rate_func, smooth/linear — always relevant

Tier 2 conditional rules (~150-300 tokens each, triggered by keywords):
  positioning.md        — move_to, next_to, align_to, shift
  grouping.md           — VGroup, arrange, layout
  latex.md              — MathTex, raw strings, t2c color mapping
  axes.md               — NumberPlane, Axes, coordinate systems
  updaters.md           — ValueTracker, always-on updaters
  transform-animations.md — Transform, ReplacementTransform, morphing
  animation-groups.md   — AnimationGroup, LaggedStart, Succession
  text-animations.md    — Write, AddTextLetterByLetter, TypeWithCursor
  colors.md             — Color constants, gradients
  styling.md            — Fill, stroke, opacity

Tier 3 skipped (overlap with our pipeline rules):
  config.md     — we own manim.cfg portrait rendering
  cli.md        — we own the manim command builder in renderer.py
  3d.md         — not used in ProcEx currently
  camera.md     — our MovingCameraScene rules in LAYOUT_RULES cover this
  scenes.md     — our SCRIPT_SYSTEM covers scene structure
  mobjects.md   — our zone contract covers positioning philosophy
  shapes.md     — our layout rules cover primitive sizing
  lines.md      — our zone contract covers arrow/line placement
  text.md       — merged into latex.md coverage
"""
from __future__ import annotations
import os
from typing import Optional


# ── File locations ─────────────────────────────────────────────────────────────
_RULES_DIR = os.path.join(
    os.path.dirname(__file__),   # utils/
    "..",                         # procex root
    "skills", "manim_skill", "rules"
)

# ── Tier 1: always injected ────────────────────────────────────────────────────
TIER1_FILES = [
    "animations.md",
    "creation-animations.md",
    "timing.md",
]

# ── Tier 2: keyword → file mapping ────────────────────────────────────────────
# Each entry: (file, [trigger_keywords])
# Keywords are checked against scene.visual_prompt.lower() + narration.lower()
#
# Tier 3 — never selected (we own these better):
#   text.md    — covered by latex.md
#   cli.md     — we own the manim command builder in renderer.py
#   config.md  — we own manim.cfg portrait rendering in renderer.py
TIER2_MAP = [
    # ── Positioning & Layout ─────────────────────────────────────────────────
    ("positioning.md",          ["move_to", "next_to", "align", "shift", "position",
                                  "place", "layout", "overlap", "crowd"]),
    ("grouping.md",             ["vgroup", "group", "arrange", "stack", "align",
                                  "column", "row", "grid", "layout"]),

    # ── Math & Text ──────────────────────────────────────────────────────────
    ("latex.md",                ["mathtex", "tex", "latex", "formula", "equation",
                                  "math", "fraction", "integral", "sum", "product",
                                  "symbol", "notation", "proof", "derivation",
                                  "matrix", "softmax", "weight", "gradient",
                                  "function", "score", "probability", "loss"]),
    ("text-animations.md",      ["typewriter", "type", "letter by letter", "cursor",
                                  "addtext", "typing effect"]),

    # ── Coordinate Systems & Graphing ────────────────────────────────────────
    ("axes.md",                 ["numberplane", "axes", "coordinate", "graph",
                                  "plot", "function", "curve", "axis", "origin",
                                  "vector space", "plane", "cartesian"]),

    # ── 3D — triggered by depth, rotation, 3D objects, or ML visualisations ─
    # e.g. attention heads, neural net layers, sphere projections,
    # rotating matrices, orbital mechanics, manifolds, embedding spaces
    ("3d.md",                   ["3d", "three", "threedscene", "sphere", "surface",
                                  "rotate", "rotation", "z-axis", "depth", "cube",
                                  "cylinder", "torus", "prism", "perspective",
                                  "attention", "transformer", "layer", "head",
                                  "neural", "orbit", "manifold", "embedding space",
                                  "high-dimensional", "dimension", "project onto",
                                  "hyperplane", "latent space"]),

    # ── Camera ───────────────────────────────────────────────────────────────
    # We cover WHEN to use MovingCameraScene; this covers HOW
    ("camera.md",               ["zoom", "pan", "camera", "focus", "follow",
                                  "track", "dolly", "frame", "movingcamera",
                                  "scale frame", "move frame", "cinematic"]),

    # ── Geometry ─────────────────────────────────────────────────────────────
    ("shapes.md",               ["polygon", "hexagon", "pentagon", "star",
                                  "rounded", "regular", "triangle", "diamond",
                                  "bounding", "irregular", "roundedrectangle"]),

    # ── Lines & Connectors ───────────────────────────────────────────────────
    ("lines.md",                ["arrow", "vector", "dashed", "brace", "connect",
                                  "connector", "link", "edge", "path", "curved arrow",
                                  "double arrow", "annotation line", "line", "segment",
                                  "parallel", "perpendicular", "transversal"]),

    # ── Dynamics ─────────────────────────────────────────────────────────────
    ("updaters.md",             ["valuetracker", "updater", "dynamic", "track",
                                  "animate value", "continuous", "slider",
                                  "real-time", "live update"]),

    # ── Transformation ───────────────────────────────────────────────────────
    ("transform-animations.md", ["transform", "morph", "replace", "change into",
                                  "becomes", "convert", "replacement", "transition"]),
    ("animation-groups.md",     ["animationgroup", "laggedstart", "succession",
                                  "stagger", "sequence", "one by one", "cascade",
                                  "reveal", "lag", "simultaneous"]),

    # ── Visual Styling ───────────────────────────────────────────────────────
    ("colors.md",               ["gradient", "color map", "interpolate_color",
                                  "set_color_by", "hue", "spectrum", "rainbow",
                                  "colour", "palette"]),
    ("styling.md",              ["opacity", "fill", "stroke", "transparency",
                                  "border", "outline", "glow", "shadow", "blur"]),

    # ── Scenes & Mobjects ────────────────────────────────────────────────────
    ("scenes.md",               ["setup", "teardown", "section", "transition",
                                  "clear screen", "multiple scenes", "scene type"]),
    ("mobjects.md",             ["custom", "vmobject", "subclass", "inherit",
                                  "custom shape", "custom animation", "bezier",
                                  "control point"]),
]

def _load_file(filename: str) -> Optional[str]:
    """Read a rule file. Returns None if file doesn't exist."""
    path = os.path.normpath(os.path.join(_RULES_DIR, filename))
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def _file_exists(filename: str) -> bool:
    return os.path.exists(os.path.normpath(os.path.join(_RULES_DIR, filename)))


def select(
    visual_prompt:  str,
    narration_text: str = "",
    element_count:  int = 0,
    verbose:        bool = False,
) -> str:
    """
    Select and return the relevant rule file content for a scene.

    Parameters
    ----------
    visual_prompt   : Scene's visual_prompt string (main keyword source).
    narration_text  : Scene's narration — secondary keyword source.
    element_count   : Number of elements — many elements → inject grouping.md.
    verbose         : Log which files were selected.

    Returns
    -------
    Formatted string ready to inject into the ManimCoder prompt.
    If no rules directory exists, returns empty string gracefully.
    """
    if not os.path.isdir(os.path.normpath(_RULES_DIR)):
        # Rules not installed — fail silently, pipeline continues unaffected
        return ""

    search_text = (visual_prompt + " " + narration_text).lower()

    # ── Tier 1 — always ───────────────────────────────────────────────────────
    selected   = list(TIER1_FILES)

    # ── Tier 2 — conditional ──────────────────────────────────────────────────
    for filename, keywords in TIER2_MAP:
        if filename in selected:
            continue
        if any(kw in search_text for kw in keywords):
            selected.append(filename)

    # Extra heuristic: many elements → grouping is almost always needed
    if element_count >= 4 and "grouping.md" not in selected:
        selected.append("grouping.md")

    # ── Load content ──────────────────────────────────────────────────────────
    blocks  = []
    missing = []
    loaded  = []

    for fname in selected:
        content = _load_file(fname)
        if content:
            blocks.append(f"### {fname}\n{content}")
            loaded.append(fname)
        else:
            missing.append(fname)

    if not blocks:
        return ""

    if verbose:
        print(f"[ManimSkillSelector] Loaded: {loaded}")
        if missing:
            print(f"[ManimSkillSelector] Missing: {missing}")

    header = (
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  MANIM CE BEST PRACTICES  (battle-tested API reference)     ║\n"
        "║  Use these patterns — they are known to work correctly.      ║\n"
        "║  Where these conflict with layout rules above, layout wins.  ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
    )

    return header + "\n\n".join(blocks)


def available_rules() -> list[str]:
    """Return list of rule files that exist on disk."""
    if not os.path.isdir(os.path.normpath(_RULES_DIR)):
        return []
    return sorted(f for f in os.listdir(os.path.normpath(_RULES_DIR))
                  if f.endswith(".md"))