"""
agents/base_agent.py — Abstract base for all ProcEx agents.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from state import ProcExState
from config import ProcExConfig
from utils.llm_client import LLMClient


class BaseAgent(ABC):
    name: str = "BaseAgent"

    def __init__(self, cfg: ProcExConfig, llm: LLMClient):
        self.cfg = cfg
        self.llm = llm

    @abstractmethod
    def run(self, state: ProcExState) -> ProcExState:
        """Execute this agent. Mutates and returns state."""
        ...

    def _log(self, msg: str) -> None:
        print(f"[{self.name}] {msg}")

    def _err(self, state: ProcExState, msg: str) -> None:
        state.log_error(self.name, msg)
