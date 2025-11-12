import json
import os
import torch
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

_MAX_TOKEN_ = 4000

ARCHITECT_SYSTEM_PROMPT = """You are Matrioska Architect. Decompose user requests into isolated modules that communicate via shared_state.

RULES:
1. Each module can READ from shared_state (dependencies)
2. Each module can WRITE to shared_state (outputs)
3. Specify exactly what each module reads/writes
4. CODE GUIDELINES:
   - Keep implementation concepts SIMPLE and CLEAN
   - Prefer direct, minimal solutions over complex ones
   - You may suggest using CDNs (Bootstrap, SweetAlert, etc.)
   - You may suggest libraries (Flask, SQLAlchemy, etc.) listed in requirements.txt or similar if relevant
   - Avoid complex build systems unless strictly required
   - Include only essential dependencies

EXAMPLE:
Request: "Library management system with dashboard"
Output: {
  "project_name": "Library Management System",
  "general_manual": {
    "goal": "Create a complete library management system with authentication, CRUD, and dashboard",
    "modules": [
      {
        "id": "html_structure",
        "name": "HTML Structure",
        "description": "Design all HTML pages and navigation layout",
        "inputs": "design requirements",
        "outputs": "HTML page structure with element IDs",
        "dependencies": [],
        "rules": "Use semantic HTML5 and IDs. Keep structure simple. You may suggest Bootstrap via CDN for layout.",
        "shared_state_reads": [],
        "shared_state_writes": ["element_ids", "page_structure"]
      },
      {
        "id": "css_styling",
        "name": "CSS Styling",
        "description": "Define visual style for all pages",
        "inputs": "HTML structure",
        "outputs": "CSS file or theme reference",
        "dependencies": ["html_structure"],
        "rules": "Use element_ids from shared_state. May suggest Bootstrap themes or custom CSS. Keep style cohesive and simple.",
        "shared_state_reads": ["element_ids", "page_structure"],
        "shared_state_writes": ["css_classes", "color_scheme"]
      },
      {
        "id": "auth_logic",
        "name": "Authentication Logic",
        "description": "Login/logout using localStorage or backend API",
        "inputs": "HTML IDs",
        "outputs": "Auth logic plan",
        "dependencies": ["html_structure"],
        "rules": "Use element_ids. May suggest SweetAlert for alerts. Keep logic lightweight and avoid heavy frameworks.",
        "shared_state_reads": ["element_ids"],
        "shared_state_writes": ["auth_api", "storage_keys"]
      },
      {
        "id": "backend_api",
        "name": "Backend API (Python)",
        "description": "Define backend API endpoints and dependencies",
        "inputs": "functional requirements",
        "outputs": "API contract and dependency list",
        "dependencies": ["auth_logic"],
        "rules": "Propose a simple Flask or FastAPI backend. List dependencies in requirements.txt (e.g. Flask, SQLAlchemy).",
        "shared_state_reads": ["auth_api"],
        "shared_state_writes": ["api_routes", "db_models"]
      }
    ],
    "integration_rules": "Combine all modules ensuring consistent shared_state keys. Use CDNs in HTML suggestions and requirements.txt for backend dependencies."
  },
  "specific_manuals": [
    {
      "module_id": "html_structure",
      "manual_text": "Define HTML structure for login, catalog, and dashboard pages. Use IDs like #loginForm, #bookList, #dashboardStats. Mention that Bootstrap via CDN may be used for layout consistency. Extract all IDs and write them to shared_state as 'element_ids'."
    }
  ]
}

NOW PROCESS THIS REQUEST:"""

class LocalLLM:
    """Local LLM for T4 (16GB VRAM)"""

    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        print(f"🔄 Loading {model_name}...")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True
        )

        self.model_name = model_name
        print(f"✅ Model loaded: {model_name}")
        print(f"💾 VRAM used: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    def generate(self, prompt: str, max_tokens: int = _MAX_TOKEN_, system: str = "") -> str:
        """Optimized generation"""
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        else:
            full_prompt = prompt

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                top_p=0.85,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        log_dir = "/content/log"
        log_file = os.path.join(log_dir, "log.txt")
        os.makedirs(log_dir, exist_ok=True)
        with open(log_file, "a") as f:
            f.write("PROMPT:\n==========================================\n\n")
            f.write(full_prompt)
            f.write("\n==========================================\n")
            f.write("RESULT:\n==========================================\n\n")
            f.write(response.strip())
            f.write("\n\n\n\n")

        return response.strip()

@dataclass
class ModuleSpec:
    id: str
    name: str
    description: str
    inputs: str
    outputs: str
    dependencies: List[str]
    rules: str
    shared_state_reads: List[str] = field(default_factory=list)
    shared_state_writes: List[str] = field(default_factory=list)

@dataclass
class GeneralManual:
    goal: str
    modules: List[ModuleSpec]
    integration_rules: str

@dataclass
class SpecificManual:
    module_id: str
    manual_text: str

@dataclass
class Architecture:
    project_name: str
    general_manual: GeneralManual
    specific_manuals: List[SpecificManual]

@dataclass
class ModuleArtifact:
    module_id: str
    name: str
    content: str
    shared_state_updates: Dict[str, Any] = field(default_factory=dict)

class ContextManager:
    def __init__(self, base_path: str = "/content"):
        self.base_path = base_path
        self.artifacts_dir = os.path.join(base_path, "matrioska_artifacts")
        self.checkpoints_dir = os.path.join(base_path, "matrioska_checkpoints")
        self.shared_state_file = os.path.join(self.checkpoints_dir, "shared_state.json")

        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.shared_state: Dict[str, Any] = self._load_shared_state()

    def _load_shared_state(self) -> Dict[str, Any]:
        """Load shared state"""
        try:
            with open(self.shared_state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
                print("🧠 [SHARED STATE] Context recovered")
                return state
        except FileNotFoundError:
            print("🧠 [SHARED STATE] Starting new context")
            return {}

    def _save_shared_state(self):
        """Persist shared state"""
        with open(self.shared_state_file, "w", encoding="utf-8") as f:
            json.dump(self.shared_state, f, indent=2, ensure_ascii=False)

    def update_shared_state(self, updates: Dict[str, Any]):
        """Update whiteboard"""
        self.shared_state.update(updates)
        self._save_shared_state()
        print(f"🧠 [SHARED STATE] Updated: {list(updates.keys())}")

    def get_shared_context(self, keys: List[str]) -> Dict[str, Any]:
        """Retrieve relevant context for module"""
        context = {}
        for key in keys:
            if key in self.shared_state:
                context[key] = self.shared_state[key]
        return context

    def save_architecture(self, arch: Architecture):
        filepath = os.path.join(self.checkpoints_dir, "architecture.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "project_name": arch.project_name,
                "general_manual": {
                    "goal": arch.general_manual.goal,
                    "modules": [asdict(m) for m in arch.general_manual.modules],
                    "integration_rules": arch.general_manual.integration_rules
                },
                "specific_manuals": [asdict(m) for m in arch.specific_manuals]
            }, f, indent=2, ensure_ascii=False)
        print(f"💾 [CHECKPOINT] Architecture → {filepath}")

    def load_architecture(self) -> Optional[Architecture]:
        filepath = os.path.join(self.checkpoints_dir, "architecture.json")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            modules = [ModuleSpec(**m) for m in data["general_manual"]["modules"]]
            general = GeneralManual(
                goal=data["general_manual"]["goal"],
                modules=modules,
                integration_rules=data["general_manual"]["integration_rules"]
            )
            specific = [SpecificManual(**m) for m in data["specific_manuals"]]

            print("📂 [RESTORATION] Context recovered")
            return Architecture(
                project_name=data["project_name"],
                general_manual=general,
                specific_manuals=specific
            )
        except FileNotFoundError:
            return None

    def save_artifact(self, artifact: ModuleArtifact):
        filename = os.path.join(self.artifacts_dir, f"{artifact.module_id}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(artifact.content)
        print(f"💾 {artifact.name} → {filename}")

    def get_artifacts_path(self) -> str:
        return self.artifacts_dir

class MatrioskaOrchestrator:
    def __init__(self, llm: LocalLLM, base_path: str = "/content"):
        self.llm = llm
        self.context_manager = ContextManager(base_path)

    def run(self, task: str, verbose: bool = True):
        """Main pipeline with SharedState"""
        if verbose:
            print("\n" + "="*80)
            print("🪆 matrioska ORCHESTRATOR - Hyperfocus + SharedState")
            print("="*80 + "\n")

        if verbose:
            print("🏗️  PHASE 1: ARCHITECTURE")
            print("-" * 80)
        architecture = self._architecture_phase(task, verbose)

        if not architecture:
            return None

        self.context_manager.save_architecture(architecture)

        if verbose:
            print("\n⚡ PHASE 2: EXECUTION (Hyperfocus + Communication)")
            print("-" * 80)
        artifacts = self._execution_phase(architecture, verbose)

        if verbose:
            print("\n🔧 PHASE 3: ASSEMBLY")
            print("-" * 80)
        restored_arch = self.context_manager.load_architecture()
        final_result = self._assembly_phase(restored_arch, artifacts, verbose)

        if verbose:
            print("\n✅ FINAL RESULT")
            print("=" * 80)
            self._display_results(architecture, artifacts, final_result)
            print("=" * 80)
            print(f"\n📁 Artifacts: {self.context_manager.get_artifacts_path()}")
            print(f"🧠 SharedState: {self.context_manager.shared_state_file}")

        return {
            "architecture": architecture,
            "artifacts": artifacts,
            "final_result": final_result,
            "shared_state": self.context_manager.shared_state
        }

    def _architecture_phase(self, task: str, verbose: bool) -> Optional[Architecture]:
        """Architectural decomposition"""
        if verbose:
            print(f"📋 Task: '{task}'")

        response = self.llm.generate(task, max_tokens=_MAX_TOKEN_, system=ARCHITECT_SYSTEM_PROMPT)

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("JSON not found")

            json_str = response[start:end]
            data = json.loads(json_str)

            modules = [ModuleSpec(**m) for m in data["general_manual"]["modules"]]
            general = GeneralManual(
                goal=data["general_manual"]["goal"],
                modules=modules,
                integration_rules=data["general_manual"]["integration_rules"]
            )
            specific = [SpecificManual(**m) for m in data["specific_manuals"]]

            arch = Architecture(
                project_name=data["project_name"],
                general_manual=general,
                specific_manuals=specific
            )

            if verbose:
                print(f"\n✓ Project: {arch.project_name}")
                print(f"✓ Goal: {arch.general_manual.goal}")
                print(f"✓ Modules: {len(arch.general_manual.modules)}")
                for i, m in enumerate(arch.general_manual.modules, 1):
                    reads = f" 📖{m.shared_state_reads}" if m.shared_state_reads else ""
                    writes = f" ✍️{m.shared_state_writes}" if m.shared_state_writes else ""
                    print(f"   {i}. {m.name}{reads}{writes}")

            return arch

        except Exception as e:
            if verbose:
                print(f"⚠️  Parsing failed, using fallback: {e}")
            return self._create_fallback_architecture(task)

    def _create_fallback_architecture(self, task: str) -> Architecture:
        """Simplified architecture"""
        return Architecture(
            project_name=f"Project: {task[:30]}",
            general_manual=GeneralManual(
                goal=task,
                modules=[
                    ModuleSpec(
                        id="mod_main",
                        name="Implementation",
                        description=task,
                        inputs="Requirements",
                        outputs="Solution",
                        dependencies=[],
                        rules="Be comprehensive"
                    )
                ],
                integration_rules="Return result"
            ),
            specific_manuals=[
                SpecificManual(
                    module_id="mod_main",
                    manual_text=f"Execute: {task}"
                )
            ]
        )

    def _execution_phase(self, architecture: Architecture, verbose: bool) -> List[ModuleArtifact]:
        artifacts = []
        manuals_map = {m.module_id: m for m in architecture.specific_manuals}

        for module_spec in architecture.general_manual.modules:
            if verbose:
                print(f"\n🎯 {module_spec.name}")

            context = self.context_manager.get_shared_context(module_spec.shared_state_reads)

            if context and verbose:
                print(f"   📖 Reading context: {list(context.keys())}")

            manual = manuals_map.get(module_spec.id)
            manual_text = manual.manual_text if manual else module_spec.description

            context_text = ""
            if context:
                context_text = "\n\nAVAILABLE CONTEXT (from previous modules):\n"
                for key, value in context.items():
                    context_text += f"- {key}: {json.dumps(value, ensure_ascii=False)}\n"

            prompt = f"""MODULE: {module_spec.name}

MANUAL:
{manual_text}{context_text}

RULES:
{module_spec.rules}

Execute this task. If you generate data that other modules need, list them at the end in the format:
SHARED_STATE_UPDATE:
{{
  "key1": "value1",
  "key2": ["item1", "item2"]
}}
"""

            content = self.llm.generate(prompt, max_tokens=_MAX_TOKEN_)

            updates = self._extract_shared_state_updates(content)

            if updates:
                self.context_manager.update_shared_state(updates)
                if verbose:
                    print(f"   ✍️ Wrote: {list(updates.keys())}")

            artifact = ModuleArtifact(
                module_id=module_spec.id,
                name=module_spec.name,
                content=content,
                shared_state_updates=updates
            )

            self.context_manager.save_artifact(artifact)
            artifacts.append(artifact)

            if verbose:
                print(f"   ✓ Generated ({len(content)} chars)")

        return artifacts

    def _extract_shared_state_updates(self, content: str) -> Dict[str, Any]:
        try:
            marker = "SHARED_STATE_UPDATE:"
            if marker in content:
                start = content.find(marker) + len(marker)
                json_start = content.find("{", start)
                if json_start != -1:
                    brace_count = 0
                    json_end = json_start
                    for i, char in enumerate(content[json_start:], json_start):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break

                    json_str = content[json_start:json_end]
                    return json.loads(json_str)
        except Exception as e:
            print(f"   ⚠️  Failed to extract shared_state: {e}")

        return {}

    def _assembly_phase(self, architecture: Architecture, artifacts: List[ModuleArtifact], verbose: bool) -> str:
        if verbose:
            print("🔗 Integrating artifacts...")

        artifacts_text = ""
        for artifact in artifacts:
            artifacts_text += f"\n{'='*60}\n{artifact.name}\n{'='*60}\n{artifact.content}\n"

        shared_state_text = json.dumps(self.context_manager.shared_state, indent=2, ensure_ascii=False)

        prompt = f"""PROJECT: {architecture.project_name}

INTEGRATION RULES:
{architecture.general_manual.integration_rules}

SHARED STATE (Contracts between modules):
{shared_state_text}

ARTIFACTS:
{artifacts_text}

Integrate the artifacts following the rules. Use the SHARED STATE to ensure that IDs, APIs and contracts are consistent."""

        return self.llm.generate(prompt, max_tokens=2000)

    def _display_results(self, arch: Architecture, artifacts: List[ModuleArtifact], final: str):
        """Formatted display"""
        print(f"\n📦 {arch.project_name}")
        print(f"🎯 {arch.general_manual.goal}")
        print(f"\n📂 Artifacts: {len(artifacts)}")
        print(f"🧠 SharedState Keys: {list(self.context_manager.shared_state.keys())}")
        print(f"\n🔗 Integrated Result:")
        print("-" * 80)
        print(final)