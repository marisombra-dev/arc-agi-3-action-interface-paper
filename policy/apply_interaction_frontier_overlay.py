from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / 'research' / 'sonpham-wm' / 'ARC3-Inference'
TOOL = TARGET / 'inference' / 'agent' / 'tool_agent.py'
SRC = ROOT / 'model_harness' / 'interaction_frontier.py'
DST = TARGET / 'inference' / 'agent' / 'interaction_frontier.py'
if not TOOL.exists() or not SRC.exists():
    raise SystemExit(f'missing interaction-frontier target under {TARGET}')
shutil.copy2(SRC, DST)
text = TOOL.read_text(encoding='utf-8')

anchor = 'from inference.agent import host_evidence as host_evidence\n'
line = 'from inference.agent.interaction_frontier import enumerate_interventions, observe_intervention, rank_interventions\n'
if line not in text:
    if text.count(anchor) != 1:
        raise RuntimeError('interaction import anchor missing')
    text = text.replace(anchor, anchor + line, 1)
init_anchor = '        self._last_action_result: dict[str, Any] | None = None\n        self._summarized_knowledge = _empty_world_model()\n'
init_new = '''        self._last_action_result: dict[str, Any] | None = None
        self._patricia_interaction_observations: list[dict[str, Any]] = []
        self._patricia_pending_intervention: dict[str, Any] | None = None
        self._patricia_current_interventions: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._summarized_knowledge = _empty_world_model()
'''
if '_patricia_interaction_observations:' not in text:
    if text.count(init_anchor) != 1:
        raise RuntimeError('interaction init anchor missing')
    text = text.replace(init_anchor, init_new, 1)

reset_anchor = '            self._last_action_result = None\n            self._summarized_knowledge = _empty_world_model()\n'
reset_new = '''            self._last_action_result = None
            self._patricia_interaction_observations = []
            self._patricia_pending_intervention = None
            self._patricia_current_interventions = {}
            self._summarized_knowledge = _empty_world_model()
'''
if '_patricia_interaction_observations = []' not in text:
    if text.count(reset_anchor) != 1:
        raise RuntimeError('interaction reset anchor missing')
    text = text.replace(reset_anchor, reset_new, 1)
method_anchor = '    def _build_user_message(self, user_prompt: str, current_frame: Frame | None) -> dict[str, Any]:\n'
method = '''    def _interaction_frontier_lines(self, current_frame: Frame | None, valid_actions: list[str] | None) -> list[str]:
        legal = list(_normalize_valid_actions(valid_actions))
        if current_frame is None or "MOUSE" not in legal:
            self._patricia_current_interventions = {}
            return []
        frontier = enumerate_interventions(current_frame.grid, legal, max_targets=16)
        ranked = rank_interventions(
            frontier,
            self._patricia_interaction_observations,
            pending_source=self._patricia_pending_intervention,
        )
        mouse_ranked = [item for item in ranked if item.get("action") == "MOUSE"][:8]
        current: dict[tuple[Any, ...], dict[str, Any]] = {}
        rendered: list[str] = []
        for item in mouse_ranked:
            row, col = int(item["row"]), int(item["col"])
            current[("MOUSE", row, col)] = dict(item)
            roles = "/".join(str(value) for value in item.get("target_roles", [])[:3]) or "visual_target"
            colors = ",".join(str(value) for value in item.get("target_colors", [])[:4]) or "-"
            size = int(item.get("target_size") or 0)
            rendered.append(f"({row},{col}) {roles} colors={colors} size={size} score={int(item.get('frontier_score') or 0)}")
        self._patricia_current_interventions = current
        if not rendered:
            return []
        lines = ["HOST_INTERACTION: ranked source-free MOUSE candidates: " + "; ".join(rendered) + "."]
        if self._patricia_pending_intervention:
            pending = self._patricia_pending_intervention
            reason = str(pending.get("_followup_reason") or "")
            source = f"({pending.get('row')},{pending.get('col')})" if pending.get("row") is not None else "previous MOUSE"
            if reason == "latent_selection":
                lines.append("HOST_INTERACTION: " + source + " is an object-like click that may have armed a latent selection despite little/no clearly causal visual change. Test a different plausible destination once before discarding that hypothesis.")
            else:
                lines.append("HOST_INTERACTION: " + source + " produced a likely local nonterminal transition; a different candidate may be a destination/follow-up target.")
        lines.append("Treat these coordinates as candidate interventions, not asserted goals. Prefer them over arbitrary coordinates unless direct visual evidence supports another point.")
        return lines

'''
if 'def _interaction_frontier_lines' not in text:
    if text.count(method_anchor) != 1:
        raise RuntimeError('interaction prompt method anchor missing')
    text = text.replace(method_anchor, method + method_anchor, 1)
prompt_anchor = '''        lines.extend(
            [
                state_line,
                f"Valid actions right now: {_format_valid_action_line(valid_actions)}.",
            ]
        )
        lines.extend(self._summarized_knowledge_lines())
'''
prompt_new = '''        lines.extend(
            [
                state_line,
                f"Valid actions right now: {_format_valid_action_line(valid_actions)}.",
            ]
        )
        lines.extend(self._interaction_frontier_lines(current_frame, valid_actions))
        _interaction_result = (self._last_action_result or {}).get("interaction_observation") if isinstance(self._last_action_result, dict) else None
        if isinstance(_interaction_result, dict):
            lines.append(
                "HOST_INTERACTION_RESULT: "
                + str(_interaction_result.get("transition_class") or "unknown")
                + f" changed_cells={int(_interaction_result.get('changed_cells') or 0)}"
                + " causality=" + str(_interaction_result.get("causality") or "unknown")
                + " followup=" + str(_interaction_result.get("followup_reason") or "none")
            )
        lines.extend(self._summarized_knowledge_lines())
'''
if 'HOST_INTERACTION_RESULT:' not in text:
    if text.count(prompt_anchor) != 1:
        raise RuntimeError('interaction prompt insertion anchor missing')
    text = text.replace(prompt_anchor, prompt_new, 1)

tool_desc_old = '"description": "Choose and execute exactly one currently valid action using the carried evidence and rules.",\n'
tool_desc_new = '"description": "Choose and execute exactly one currently valid action using the carried evidence and rules. For MOUSE, prefer a HOST_INTERACTION candidate coordinate unless direct visual evidence supports another point.",\n'
if tool_desc_new not in text:
    if text.count(tool_desc_old) != 1:
        raise RuntimeError('interaction act description anchor missing')
    text = text.replace(tool_desc_old, tool_desc_new, 1)
record_anchor = '    def _run_python_tool(self, state_path: Path, arguments: dict[str, Any]) -> _ToolDispatchResult:\n'
record_method = '''    def _record_interaction_observation(self, before_frame: Frame | None, after_frame: Frame | None, actions: list[dict[str, Any]], compact_payload: dict[str, Any]) -> None:
        if before_frame is None or after_frame is None or len(actions) != 1:
            return
        request = actions[0]
        action = str(request.get("action") or "").strip().upper()
        if not action:
            return
        row = request.get("row") if action == "MOUSE" else None
        col = request.get("col") if action == "MOUSE" else None
        key = (action, int(row), int(col)) if isinstance(row, int) and isinstance(col, int) else (action, None, None)
        intervention = dict(self._patricia_current_interventions.get(key) or {})
        if not intervention:
            intervention = {
                "id": "EXEC",
                "action": action,
                "row": row,
                "col": col,
                "target_id": None,
                "intervention_class": f"control:{action}" if action != "MOUSE" else "control:MOUSE->free_coordinate",
            }
        observation = observe_intervention(
            before_frame.grid,
            after_frame.grid,
            intervention,
            reward=float(compact_payload.get("reward") or 0.0),
            level_completed=bool(compact_payload.get("level_completed")),
            terminal=bool(compact_payload.get("game_over") or compact_payload.get("run_complete") or compact_payload.get("done")),
        )
        observation["level"] = compact_payload.get("level")
        self._patricia_interaction_observations.append(observation)
        self._patricia_interaction_observations = self._patricia_interaction_observations[-64:]
'''
if 'def _record_interaction_observation' not in text:
    if text.count(record_anchor) != 1:
        raise RuntimeError('interaction observation method anchor missing')
    text = text.replace(record_anchor, record_method + record_anchor, 1)
record_tail = '''        compact_payload["interaction_observation"] = {
            "transition_class": observation.get("transition_class"),
            "changed_cells": observation.get("changed_cells"),
            "scope": observation.get("scope"),
            "motions": observation.get("motions"),
            "intervention_class": observation.get("intervention_class"),
            "intervention_key": observation.get("intervention_key"),
            "target_id": observation.get("target_id"),
            "causality": observation.get("causality"),
            "latent_selection": observation.get("latent_selection"),
            "followup_reason": observation.get("followup_reason"),
        }
        if action == "MOUSE" and observation.get("followup_warranted") and not compact_payload.get("level_completed") and not compact_payload.get("game_over") and not compact_payload.get("run_complete"):
            pending = dict(intervention)
            pending["intervention_key"] = observation.get("intervention_key")
            pending["_followup_reason"] = observation.get("followup_reason")
            pending["_causality"] = observation.get("causality")
            self._patricia_pending_intervention = pending
        else:
            self._patricia_pending_intervention = None

'''
marker = '        self._patricia_interaction_observations = self._patricia_interaction_observations[-64:]\n'
if 'compact_payload["interaction_observation"]' not in text:
    if text.count(marker) != 1:
        raise RuntimeError('interaction observation tail anchor missing')
    text = text.replace(marker, marker + record_tail, 1)

before_anchor = '            normalized_actions = self._normalize_python_actions(actions)\n'
before_new = '''            normalized_actions = self._normalize_python_actions(actions)
            try:
                _interaction_before_frame, _ = load_runtime_state(state_path)
            except Exception:
                _interaction_before_frame = None
'''
if '_interaction_before_frame' not in text:
    if text.count(before_anchor) != 1:
        raise RuntimeError('interaction before-frame anchor missing')
    text = text.replace(before_anchor, before_new, 1)
after_anchor = '''            if isinstance(next_valid_actions, list):
                self._current_valid_actions = _normalize_valid_actions(next_valid_actions)
            if compact_payload.get("executed") and _terminal_action_reason(compact_payload):
'''
after_new = '''            if isinstance(next_valid_actions, list):
                self._current_valid_actions = _normalize_valid_actions(next_valid_actions)
            try:
                _interaction_after_frame, _ = load_runtime_state(state_path)
            except Exception:
                _interaction_after_frame = None
            self._record_interaction_observation(
                _interaction_before_frame,
                _interaction_after_frame,
                normalized_actions,
                compact_payload,
            )
            if compact_payload.get("executed") and _terminal_action_reason(compact_payload):
'''
if 'self._record_interaction_observation(' not in text:
    if text.count(after_anchor) != 1:
        raise RuntimeError('interaction after-frame anchor missing')
    text = text.replace(after_anchor, after_new, 1)

TOOL.write_text(text, encoding='utf-8')
print(f'interaction-frontier overlay applied to {TARGET}')
