from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / 'research' / 'sonpham-wm' / 'ARC3-Inference'
TOOL = TARGET / 'inference' / 'agent' / 'tool_agent.py'
if not TOOL.exists():
    raise SystemExit(f'missing tool_agent.py under {TARGET}')
text = TOOL.read_text(encoding='utf-8')
old = '                        "required": (["action", "candidate_index", "world_model"] if legal == ["MOUSE"] else ["action", "world_model"]),\n'
new = '                        "required": (["action", "candidate_index", "world_model"] if legal == ["MOUSE"] and bool(getattr(self, "_patricia_current_interventions", {})) else ["action", "world_model"]),\n'
if new not in text:
    if text.count(old) != 1:
        raise RuntimeError(f'mouse candidate fallback anchor count={text.count(old)}')
    text = text.replace(old, new, 1)
TOOL.write_text(text, encoding='utf-8')
print(f'mouse candidate schema fallback applied to {TARGET}')