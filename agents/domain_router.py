"""
agents/domain_router.py
Classifies the input document into a domain and loads the corresponding skill YAML.
"""
from __future__ import annotations
import yaml
import os
from pathlib import Path
from state import ProcExState, Domain
from config import ProcExConfig
from utils.llm_client import LLMClient
from agents.base_agent import BaseAgent


SKILL_FILES = {
    Domain.ML_MATH:       "skills/ml_math.yaml",
    Domain.MEDICAL:       "skills/medical_anatomy.yaml",
    Domain.CS_SYSTEMS:    "skills/cs_systems.yaml",
    Domain.NCLEX_NURSING: "skills/nclex_nursing.yaml",
    Domain.HYBRID:        "skills/medical_anatomy.yaml",  # HYBRID defaults to medical
}

DOMAIN_SYSTEM = """You are a content domain classifier for an educational video pipeline.
Classify the given text into exactly ONE of these domains:

ML_MATH       — Machine learning, deep learning, statistics, linear algebra, calculus,
                optimization, neural networks, transformers, diffusion models, etc.

MEDICAL       — Anatomy, physiology, pathophysiology, pharmacology, biochemistry,
                clinical medicine, radiology, surgery — but NOT primarily NCLEX exam prep.

CS_SYSTEMS    — Algorithms, data structures, operating systems, networking, databases,
                distributed systems, compilers, computer architecture.

NCLEX_NURSING — NCLEX exam preparation, nursing interventions, clinical reasoning for nurses,
                priority setting, medication administration from a nursing perspective.

HYBRID        — The document spans multiple domains equally (rare).

Respond with ONLY valid JSON: {"domain": "<DOMAIN>", "confidence": 0.0-1.0, "reasoning": "..."}
"""


class DomainRouter(BaseAgent):
    name = "DomainRouter"

    def run(self, state: ProcExState) -> ProcExState:
        self._log("Classifying input domain...")

        # Truncate input for classification (don't need full text)
        preview = state.raw_input[:4000]

        result = self.llm.complete_json(
            DOMAIN_SYSTEM,
            f"Classify this content:\n\n{preview}",
            temperature=0.1,
        )

        domain_str = result.get("domain", "ML_MATH")
        try:
            state.domain = Domain(domain_str)
        except ValueError:
            self._log(f"Unknown domain '{domain_str}', defaulting to ML_MATH")
            state.domain = Domain.ML_MATH

        self._log(f"Domain: {state.domain.value} (confidence={result.get('confidence', '?')})")
        self._log(f"Reasoning: {result.get('reasoning', '')}")

        # Load skill pack YAML
        # Search relative to this file (agents/) → go up one level to project root
        skill_rel = SKILL_FILES.get(state.domain, SKILL_FILES[Domain.ML_MATH])
        project_root = Path(__file__).resolve().parent.parent

        candidates = [
            project_root / skill_rel,           # <project_root>/skills/xxx.yaml
            Path.cwd() / skill_rel,             # cwd/skills/xxx.yaml (if running from project root)
        ]

        for path in candidates:
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    state.skill_pack = yaml.safe_load(f)
                self._log(f"Skill pack loaded: {path}")
                break
        else:
            self._log(f"WARNING: skill file not found, using empty pack. Tried: {candidates}")
            state.skill_pack = {}

        return state
