"""
Multi-model agent roles (§4.2 of the plan).

Each agent role has a specialized model assignment:
  Architect → reasoning-strong (Opus 4 / GPT-4.5)
  Generator → balanced (Sonnet 4 / GPT-4o)
  Validator → cheap/fast (Haiku 4.5 / GPT-4o-mini)
  Judge     → analytical precision (Sonnet 4)
  Repairer  → debugging focus (same as Generator)
  Reflector → meta-cognitive (same as Judge)
"""

from matrioska.agents.architect import ArchitectAgent
from matrioska.agents.generator import GeneratorAgent
from matrioska.agents.validator import ValidatorAgent
from matrioska.agents.judge import JudgeAgent
from matrioska.agents.repairer import RepairerAgent
from matrioska.agents.reflector import ReflectorAgent
from matrioska.agents.test_designer import TestDesignerAgent

__all__ = [
    "ArchitectAgent",
    "GeneratorAgent",
    "ValidatorAgent",
    "JudgeAgent",
    "RepairerAgent",
    "ReflectorAgent",
    "TestDesignerAgent",
]
