"""Wrapper for direct single-prompt generation via Groq-compatible API.

Sends the task prompt in a single API call and extracts files from the response.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any


def _get_client():
    """Get an OpenAI-compatible client pointed at Groq."""
    from openai import OpenAI
    return OpenAI(
        api_key=os.environ.get("MATRIOSKA_API_KEY", os.environ.get("GROQ_API_KEY", "")),
        base_url=os.environ.get("MATRIOSKA_BASE_URL", "https://api.groq.com/openai/v1"),
    )


SYSTEM_PROMPT = """You are a software engineer generating files for a coding task.
Write complete, production-ready files.
Output each file in this format:

===FILE: <filepath>===
<file content>
===END FILE===

Generate ALL requested files. Every file must be complete and syntactically valid.
Do NOT use markdown code fences inside the file content. Write raw code directly."""


def run_direct(
    task_prompt: str,
    model: str,
    api_key: str,
    api_base: str,
    work_dir: Path,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Run direct single-prompt generation.

    Returns dict with: files (list of Paths), tokens_used, duration_s, raw_response.
    """
    client = _get_client()
    client.api_key = api_key
    client.base_url = api_base

    full_prompt = f"""{SYSTEM_PROMPT}

TASK:
{task_prompt}

Generate all files now. Write each file with the ===FILE: path=== header format."""

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    duration = time.time() - start

    content = response.choices[0].message.content or ""
    usage = response.usage

    files = _extract_files(content, work_dir)

    return {
        "files": files,
        "tokens_used": usage.total_tokens if usage else 0,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "duration_s": duration,
        "raw_response": content,
        "orchestrator": "direct",
    }


FILE_PATTERN = re.compile(r'===FILE:\s*(.+?)===\s*\n(.*?)\n===END FILE===', re.DOTALL)


def _extract_files(content: str, work_dir: Path) -> list[Path]:
    """Extract files from the response content."""
    files = []

    # Try ===FILE: path=== format
    for match in FILE_PATTERN.finditer(content):
        filepath = match.group(1).strip()
        file_content = match.group(2).strip()
        # Strip markdown code fences if present
        file_content = _strip_fences(file_content)
        dest = work_dir / filepath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(file_content)
        files.append(dest)

    if files:
        return files

    # Fallback: try ```language:filepath ... ``` format
    fence_pattern = re.compile(r'```(?:\w+)?:?(.+?)\n(.*?)```', re.DOTALL)
    for match in fence_pattern.finditer(content):
        header = match.group(1).strip()
        code = match.group(2).strip()
        # Header might be a filename or language
        if '/' in header or '.' in header:
            dest = work_dir / header
        else:
            # Guess extension from language
            ext_map = {"python": "py", "javascript": "js", "html": "html", "css": "css",
                       "yaml": "yml", "json": "json", "bash": "sh", "sql": "sql"}
            ext = ext_map.get(header.lower(), "txt")
            dest = work_dir / f"output.{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(code)
        files.append(dest)

    # Fallback: write entire response as output.txt
    if not files:
        dest = work_dir / "output.txt"
        dest.write_text(content)
        files.append(dest)

    return files


def _strip_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    lines = text.strip().split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
