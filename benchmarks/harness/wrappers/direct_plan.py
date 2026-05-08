"""Wrapper for two-phase direct generation: plan then generate.

Phase 1: Generate a plan (list of files with descriptions).
Phase 2: Generate each file based on the plan.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any


def _get_client():
    from openai import OpenAI
    return OpenAI(
        api_key=os.environ.get("MATRIOSKA_API_KEY", os.environ.get("GROQ_API_KEY", "")),
        base_url=os.environ.get("MATRIOSKA_BASE_URL", "https://api.groq.com/openai/v1"),
    )


PLAN_PROMPT = """You are a software architect. Given a coding task, produce a plan as JSON.

Output ONLY a JSON object with this exact structure:
{
  "project_name": "string",
  "files": [
    {
      "path": "relative/path/to/file.ext",
      "description": "what this file should contain",
      "dependencies": ["other_file_paths"]
    }
  ]
}

Rules:
- Every file must be syntactically valid and complete
- List files in dependency order (files with no deps first)
- Be specific about what each file must contain (classes, functions, APIs)
"""

GENERATE_PROMPT = """You are a software engineer writing a specific file for a project.

PROJECT PLAN:
{plan_summary}

FILE TO WRITE: {filepath}
DESCRIPTION: {description}
DEPENDENCIES: {dependencies}

Write ONLY the raw file content. No markdown fences, no explanations.
The content must be complete and syntactically valid.
"""


def run_direct_plan(
    task_prompt: str,
    model: str,
    api_key: str,
    api_base: str,
    work_dir: Path,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    client = _get_client()
    client.api_key = api_key
    client.base_url = api_base

    total_tokens = 0
    start = time.time()

    # Phase 1: Plan
    plan_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PLAN_PROMPT},
            {"role": "user", "content": f"TASK:\n{task_prompt}\n\nProduce the JSON plan."},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    total_tokens += plan_response.usage.total_tokens if plan_response.usage else 0
    plan_text = plan_response.choices[0].message.content or ""

    # Parse plan
    plan = _parse_json(plan_text)
    files_spec = plan.get("files", [])

    # Phase 2: Generate each file sequentially
    generated_files = []
    plan_summary = json.dumps(plan, indent=2)

    for file_spec in files_spec:
        filepath = file_spec.get("path", f"output_{len(generated_files)}.txt")
        description = file_spec.get("description", "")
        dependencies = json.dumps(file_spec.get("dependencies", []))

        gen_prompt = GENERATE_PROMPT.format(
            plan_summary=plan_summary,
            filepath=filepath,
            description=description,
            dependencies=dependencies,
        )

        gen_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You write complete, production-ready code files. Output raw code only."},
                {"role": "user", "content": gen_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        total_tokens += gen_response.usage.total_tokens if gen_response.usage else 0
        content = gen_response.choices[0].message.content or ""

        dest = work_dir / filepath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(_strip_fences(content))
        generated_files.append(dest)

    duration = time.time() - start

    return {
        "files": generated_files,
        "tokens_used": total_tokens,
        "duration_s": duration,
        "plan": plan,
        "raw_response": plan_text,
        "orchestrator": "direct_plan",
    }


def _parse_json(text: str) -> dict:
    """Extract JSON from text, handling markdown fences."""
    # Remove markdown fences
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = text.strip()
    # Find JSON object boundaries
    start_idx = text.find('{')
    end_idx = text.rfind('}') + 1
    if start_idx >= 0 and end_idx > start_idx:
        text = text[start_idx:end_idx]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"files": []}


def _strip_fences(text: str) -> str:
    lines = text.strip().split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
