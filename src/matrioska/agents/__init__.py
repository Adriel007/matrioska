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
