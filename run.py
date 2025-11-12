# EXAMPLE USAGE OF MATRIOSKA ORCHESTRATOR
from matrioska_v1 import MatrioskaOrchestrator
from local_llm import LocalLLM

llm = LocalLLM("mistralai/Mistral-7B-Instruct-v0.3")
orchestrator = MatrioskaOrchestrator(llm, base_path="./")

result = orchestrator.run(input("Enter your prompt: "))

import json
with open('./matrioska_checkpoints/shared_state.json', 'r') as f:
    shared_state = json.load(f)
print(json.dumps(shared_state, indent=2))