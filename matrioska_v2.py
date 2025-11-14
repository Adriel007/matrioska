import json
import os
import torch
from json_repair import repair_json
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

_MAX_TOKEN_ = 20_000

ARCHITECT_SYSTEM_PROMPT = """You are Matrioska Architect. Decompose user requests into isolated modules that communicate via shared_state.

**OUTPUT FORMAT RULES (Mandatory):**

1.  **Structure:** The output **MUST** be a strict JSON object following the format below.
2.  **Root Object:** The root object **MUST** be named `instructs`.
3.  **Order (`order`):** Assign an integer starting from 1 to indicate the **creation order** of files (e.g., HTML/DB first, then CSS/JS/API logic).
4.  **Shared Information (`shared_state_reads/writes`):**
    * `shared_state_writes`: List **key information** this file defines (e.g., "HTML element IDs," "CSS class names," "API endpoint routes").
    * `shared_state_reads`: List **key information** this file requires from previous files in the `order` (e.g., CSS reads "HTML element IDs").
5.  **Content Requirement (`content`):** Must contain a **complete and detailed prompt** for a coding AI, demanding the generation of **full, reduced, and efficient code**. Explicitly instruct the use of **CDNs and lightweight libraries**.
6.  **Details (`details`):** Must contain a concise list of **functional and non-functional requirements**.

**Mandatory Output Structure:**

{
  "instructs": {
    "files": [
      {
        "name": "[FILE_NAME_WITHOUT_EXTENSION]",
        "extension": "[EXTENSION]",
        "order": 1,
        "shared_state_writes": ["[INFO_DEFINED_BY_THIS_FILE]"],
        "shared_state_reads": ["[INFO_NEEDED_FROM_OTHER_FILES]"],
        "content": "[DETAILED PROMPT FOR A CODING AI TO GENERATE THE ENTIRE CODE, FOCUSING ON BEING REDUCED, COMPLETE, AND USING CDNS/LIBRARIES IF NOT SPECIFIED]",
        "details": "[CONCISE FUNCTIONAL AND NON-FUNCTIONAL REQUIREMENTS]"
      }
    ]
  }
}

NOW PROCESS THIS REQUEST:"""

class LocalLLM:
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
class FileSpec:
    name: str
    extension: str
    order: int
    shared_state_writes: List[str] = field(default_factory=list)
    shared_state_reads: List[str] = field(default_factory=list)
    content: str = ""
    details: str = ""


@dataclass
class Architecture:
    project_name: str
    files: List[FileSpec]


@dataclass
class FileArtifact:
    name: str
    extension: str
    order: int
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
        try:
            with open(self.shared_state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
                print("🧠 [SHARED STATE] Context recovered")
                return state
        except FileNotFoundError:
            print("🧠 [SHARED STATE] Starting new context")
            return {}

    def _save_shared_state(self):
        with open(self.shared_state_file, "w", encoding="utf-8") as f:
            json.dump(self.shared_state, f, indent=2, ensure_ascii=False)

    def update_shared_state(self, updates: Dict[str, Any]):
        self.shared_state.update(updates)
        self._save_shared_state()
        print(f"🧠 [SHARED STATE] Updated: {list(updates.keys())}")

    def get_shared_context(self, keys: List[str]) -> Dict[str, Any]:
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
                "files": [asdict(f) for f in arch.files]
            }, f, indent=2, ensure_ascii=False)
        print(f"💾 [CHECKPOINT] Architecture → {filepath}")

    def load_architecture(self) -> Optional[Architecture]:
        filepath = os.path.join(self.checkpoints_dir, "architecture.json")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            files = [FileSpec(**f) for f in data["files"]]
            print("📂 [RESTORATION] Context recovered")
            return Architecture(
                project_name=data["project_name"],
                files=files
            )
        except FileNotFoundError:
            return None

    def save_artifact(self, artifact: FileArtifact):
        filename = os.path.join(self.artifacts_dir, f"{artifact.name}.{artifact.extension}")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(artifact.content)
        print(f"💾 {artifact.name}.{artifact.extension} → {filename}")

    def get_artifacts_path(self) -> str:
        return self.artifacts_dir


class MatrioskaOrchestrator:
    def __init__(self, llm: LocalLLM, base_path: str = "/content"):
        self.llm = llm
        self.context_manager = ContextManager(base_path)

    def run(self, task: str, verbose: bool = True):
        """Main pipeline with file-based architecture"""
        if verbose:
            print("\n" + "="*80)
            print("🪆 MATRIOSKA ORCHESTRATOR - File-Based Architecture")
            print("="*80 + "\n")

        if verbose:
            print("🏗️  PHASE 1: ARCHITECTURE")
            print("-" * 80)
        architecture = self._architecture_phase(task, verbose)

        if not architecture:
            return None

        self.context_manager.save_architecture(architecture)

        if verbose:
            print("\n⚡ PHASE 2: CODE GENERATION")
            print("-" * 80)
        artifacts = self._generation_phase(architecture, verbose)

        if verbose:
            print("\n✅ FINAL RESULT")
            print("=" * 80)
            self._display_results(architecture, artifacts)
            print("=" * 80)
            print(f"\n📁 Artifacts: {self.context_manager.get_artifacts_path()}")
            print(f"🧠 SharedState: {self.context_manager.shared_state_file}")

        return {
            "architecture": architecture,
            "artifacts": artifacts,
            "shared_state": self.context_manager.shared_state
        }

    def _architecture_phase(self, task: str, verbose: bool) -> Optional[Architecture]:
        if verbose:
            print(f"📋 Task: '{task}'")

        response = self.llm.generate(task, max_tokens=_MAX_TOKEN_, system=ARCHITECT_SYSTEM_PROMPT)

        try:
            data = json.loads(repair_json(response))

            if "instructs" not in data or "files" not in data["instructs"]:
                raise ValueError("Invalid response format: missing 'instructs.files'")

            files = [FileSpec(**f) for f in data["instructs"]["files"]]
            files.sort(key=lambda x: x.order)

            project_name = f"Project_{len(files)}_Files"
            arch = Architecture(project_name=project_name, files=files)

            if verbose:
                print(f"\n✓ Project: {arch.project_name}")
                print(f"✓ Files: {len(arch.files)}")
                for f in arch.files:
                    reads = f" 📖{f.shared_state_reads}" if f.shared_state_reads else ""
                    writes = f" ✍️{f.shared_state_writes}" if f.shared_state_writes else ""
                    print(f"   {f.order}. {f.name}.{f.extension}{reads}{writes}")

            return arch

        except Exception as e:
            if verbose:
                print(f"⚠️  Parsing failed: {e}")
                print(f"Response preview: {response[:500]}...")
            return None

    def _generation_phase(self, architecture: Architecture, verbose: bool) -> List[FileArtifact]:
        artifacts = []

        for file_spec in architecture.files:
            if verbose:
                print(f"\n🎯 Generating: {file_spec.name}.{file_spec.extension} (Order: {file_spec.order})")

            context = self.context_manager.get_shared_context(file_spec.shared_state_reads)

            if context and verbose:
                print(f"   📖 Reading context: {list(context.keys())}")

            context_text = ""
            if context:
                context_text = "\n\nAVAILABLE SHARED INFORMATION (from previous files):\n"
                for key, value in context.items():
                    context_text += f"- {key}: {json.dumps(value, ensure_ascii=False)}\n"

            prompt = f"""FILE: {file_spec.name}.{file_spec.extension}

GENERATION INSTRUCTIONS:
{file_spec.content}{context_text}

REQUIREMENTS:
{file_spec.details}

Generate the COMPLETE, REDUCED, and EFFICIENT code for this file. Use CDNs and lightweight libraries when appropriate.

If you define key information that other files need (e.g., element IDs, class names, API routes), list them at the end in the format:
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

            clean_content = self._remove_shared_state_marker(content)

            artifact = FileArtifact(
                name=file_spec.name,
                extension=file_spec.extension,
                order=file_spec.order,
                content=clean_content,
                shared_state_updates=updates
            )

            self.context_manager.save_artifact(artifact)
            artifacts.append(artifact)

            if verbose:
                print(f"   ✓ Generated ({len(clean_content)} chars)")

        return artifacts

    def _extract_shared_state_updates(self, content: str) -> Dict[str, Any]:
        try:
            marker = "SHARED_STATE_UPDATE:"
            if marker in content:
                after_marker = content[content.find(marker) + len(marker):]
                parsed = json.loads(repair_json(after_marker))

                if isinstance(parsed, list):
                    print(f"   ⚠️ SharedState as array, converting to dict")
                    return {"data_list": parsed}

                if not isinstance(parsed, dict):
                    print(f"   ⚠️ SharedState is invalid: {type(parsed).__name__}")
                    return {}

                return parsed

        except Exception as e:
            print(f"   ⚠️ Failed to extract shared_state: {e}")

        return {}

    def _remove_shared_state_marker(self, content: str) -> str:
        marker = "SHARED_STATE_UPDATE:"
        if marker in content:
            return content[:content.find(marker)].strip()
        return content

    def _display_results(self, arch: Architecture, artifacts: List[FileArtifact]):
        print(f"\n📦 {arch.project_name}")
        print(f"\n📂 Generated Files: {len(artifacts)}")
        for artifact in artifacts:
            print(f"   {artifact.order}. {artifact.name}.{artifact.extension}")
        print(f"\n🧠 SharedState Keys: {list(self.context_manager.shared_state.keys())}")