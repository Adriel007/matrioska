"""Shared text utilities for stripping markdown fences and cleaning LLM output."""
from __future__ import annotations


def strip_fences(text: str) -> str:
    """Strip markdown code fences from model output."""
    t = text.strip()
    for fence in ("```python", "```py", "```sql", "```json", "```yaml",
                  "```html", "```css", "```js", "```bash", "```sh",
                  "```dockerfile", "```yml", "```", "``"):
        if t.startswith(fence):
            t = t[len(fence):].strip()
            if t.endswith("```"):
                t = t[:-3].strip()
            break
    return t
