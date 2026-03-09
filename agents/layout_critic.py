"""
agents/layout_critic.py

Scene-level layout critic for generated Manim code.

Design goals:
- Lightweight heuristic pass to catch obvious visual-density and overlap risk.
- Optional LLM critic augmentation (text-only for now), controlled by config.
- Structured output that can be fed back into ManimCoder as retry context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re

from state import Scene, VisualStrategy
from config import ProcExConfig
from utils.llm_client import LLMClient


@dataclass
class CriticViolation:
    kind: str
    severity: str
    evidence: str


@dataclass
class CriticReport:
    passed: bool
    score: int
    violations: list[CriticViolation] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    source: str = "heuristic"


CRITIC_SYSTEM = """You are a strict UI-layout critic for Manim educational scenes.

Task:
Given scene metadata and Manim code, detect overlap/collision/readability risk.
Return ONLY JSON with this schema:
{
  "score": 0-100,
  "passed": true|false,
  "violations": [
    {"kind":"overlap|density|timing|readability", "severity":"low|medium|high", "evidence":"..."}
  ],
  "actions": ["Concrete revision instruction..."]
}

Guidelines:
- Prioritize spatial collisions, visual clutter, text readability, and insufficient content clearing.
- Be specific and code-actionable.
- If no meaningful issue exists, return passed=true with score>=85.
"""


class LayoutCritic:
    def __init__(self, cfg: ProcExConfig, llm: LLMClient):
        self.cfg = cfg
        self.llm = llm

    @staticmethod
    def applies_to(scene: Scene) -> bool:
        return scene.visual_strategy in (
            VisualStrategy.MANIM,
            VisualStrategy.TEXT_ANIMATION,
            VisualStrategy.IMAGE_MANIM_HYBRID,
        )

    def review(self, scene: Scene, code: str) -> CriticReport:
        heuristic = self._heuristic_review(scene, code)
        if not self.cfg.critic_use_llm:
            return heuristic

        try:
            llm_report = self._llm_review(scene, code, heuristic)
        except Exception:
            # Never block pipeline because critic failed.
            return heuristic

        # Conservative merge: keep all heuristic violations, blend score downward.
        merged = CriticReport(
            passed=heuristic.passed and llm_report.passed,
            score=min(heuristic.score, llm_report.score),
            violations=[*heuristic.violations, *llm_report.violations],
            actions=[*heuristic.actions, *llm_report.actions],
            source="heuristic+llm",
        )
        return merged

    def build_retry_context(self, report: CriticReport) -> str:
        if report.passed:
            return ""

        bullet_violations = "\n".join(
            f"- [{v.severity}] {v.kind}: {v.evidence}" for v in report.violations
        )
        bullet_actions = "\n".join(f"- {a}" for a in report.actions)

        return (
            "Layout critic failed this candidate. Fix ALL items before regenerating.\n"
            f"score={report.score} source={report.source}\n"
            f"Violations:\n{bullet_violations or '- none'}\n"
            f"Required actions:\n{bullet_actions or '- none'}"
        )

    def _heuristic_review(self, scene: Scene, code: str) -> CriticReport:
        violations: list[CriticViolation] = []
        actions: list[str] = []

        # Heuristic 1: dense scene risk via many constructed objects.
        object_calls = len(re.findall(r"\b(Text|MathTex|Rectangle|RoundedRectangle|Arrow|Line|Table|VGroup|Group)\(", code))
        if object_calls >= self.cfg.critic_max_objects:
            violations.append(CriticViolation(
                kind="density",
                severity="high",
                evidence=f"{object_calls} object-construction calls (threshold {self.cfg.critic_max_objects})",
            ))
            actions.append("Reduce concurrent on-screen objects; split into sequential beats with FadeOut transitions.")

        # Heuristic 2: too many manual shifts often correlates with collision-prone hand layout.
        shift_calls = len(re.findall(r"\.shift\(", code))
        arrange_calls = len(re.findall(r"\.arrange\(", code))
        if shift_calls >= self.cfg.critic_max_manual_shifts and arrange_calls == 0:
            violations.append(CriticViolation(
                kind="overlap",
                severity="medium",
                evidence=(
                    f"{shift_calls} manual .shift() calls with no .arrange(); layout likely fragile"
                ),
            ))
            actions.append("Replace manual shifts with VGroup/Group + .arrange() for collision-safe layout.")

        # Heuristic 3: entrances without clear exits increase layering clutter.
        fadein_count = len(re.findall(r"\b(FadeIn|Write|Create|GrowFromCenter)\(", code))
        fadeout_count = len(re.findall(r"\bFadeOut\(", code))
        if fadein_count > fadeout_count + self.cfg.critic_max_unbalanced_entrances:
            violations.append(CriticViolation(
                kind="timing",
                severity="high",
                evidence=f"entrance-heavy animation ({fadein_count} in vs {fadeout_count} out)",
            ))
            actions.append("Insert explicit FadeOut/ReplacementTransform before introducing new panels.")

        # Heuristic 4: unreadable long lines likely overflow when many elements coexist.
        long_text_literals = len(re.findall(r'Text\("[^"\\]{80,}"', code))
        if long_text_literals:
            violations.append(CriticViolation(
                kind="readability",
                severity="medium",
                evidence=f"{long_text_literals} very long Text(...) literals (>80 chars)",
            ))
            actions.append("Break long text into shorter bullets or reduce font and width with explicit wrapping.")

        score = 100
        for v in violations:
            score -= {"low": 8, "medium": 16, "high": 28}[v.severity]
        score = max(0, score)

        passed = score >= self.cfg.critic_pass_score and not any(v.severity == "high" for v in violations)
        return CriticReport(
            passed=passed,
            score=score,
            violations=violations,
            actions=actions,
            source="heuristic",
        )

    def _llm_review(self, scene: Scene, code: str, heuristic: CriticReport) -> CriticReport:
        violations_payload = [
            {"kind": v.kind, "severity": v.severity, "evidence": v.evidence}
            for v in heuristic.violations
        ]
        prompt = f"""Scene id: {scene.id}
Scene strategy: {scene.visual_strategy.value}
Scene duration: {scene.duration_seconds:.1f}s
Narration:\n{scene.narration_text[:1200]}

Current heuristic findings:\n{json.dumps(violations_payload, indent=2)}

Manim code:\n{code[:5000]}
"""

        raw = self.llm.complete(
            CRITIC_SYSTEM,
            prompt,
            json_mode=True,
            max_tokens=1200,
            temperature=0.2,
            primary_provider=self.cfg.critic_primary_provider,
        )
        data = json.loads(raw)

        violations = [
            CriticViolation(
                kind=str(v.get("kind", "density")),
                severity=str(v.get("severity", "medium")),
                evidence=str(v.get("evidence", "")),
            )
            for v in data.get("violations", [])
            if isinstance(v, dict)
        ]

        return CriticReport(
            passed=bool(data.get("passed", False)),
            score=int(data.get("score", 0)),
            violations=violations,
            actions=[str(a) for a in data.get("actions", [])],
            source="llm",
        )
