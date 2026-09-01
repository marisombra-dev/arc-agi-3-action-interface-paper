from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "research" / "sonpham-wm" / "ARC3-Inference"
MODE = str(sys.argv[2] if len(sys.argv) > 2 else "regime").strip().lower()
TOOL = TARGET / "inference" / "agent" / "tool_agent.py"

if MODE not in {"static", "v47", "regime"}:
    raise SystemExit(f"unsupported interaction policy mode: {MODE}")
if not TOOL.exists():
    raise SystemExit(f"missing tool_agent.py under {TARGET}")

text = TOOL.read_text(encoding="utf-8")
import_old = "from inference.agent.interaction_frontier import enumerate_interventions, observe_intervention, rank_interventions\n"
import_new = "from inference.agent.interaction_frontier import enumerate_interventions, observe_intervention, rank_interventions, rank_interventions_regime_gated\n"
if import_old in text:
    text = text.replace(import_old, import_new, 1)
elif import_new not in text:
    raise RuntimeError("interaction-frontier import anchor missing")
call_old = '''        ranked = rank_interventions(
            frontier,
            self._patricia_interaction_observations,
            pending_source=self._patricia_pending_intervention,
        )
'''
if call_old not in text:
    raise RuntimeError("interaction ranking call anchor missing")

if MODE == "v47":
    call_new = call_old
elif MODE == "regime":
    call_new = '''        ranked = rank_interventions_regime_gated(
            frontier,
            self._patricia_interaction_observations,
        )
'''
else:
    call_new = '''        ranked = [dict(item) for item in list(frontier.get("interventions") or [])]
        for item in ranked:
            item["frontier_score"] = int(item.get("priority") or 0)
'''

text = text.replace(call_old, call_new, 1)
TOOL.write_text(text, encoding="utf-8")
print(f"patched interaction policy mode={MODE} into {TOOL}")
