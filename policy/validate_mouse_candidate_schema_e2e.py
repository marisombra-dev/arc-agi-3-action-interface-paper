from __future__ import annotations

import ast
import io
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research" / "sonpham-wm" / "ARC3-Inference"
source = ast.parse((ROOT / "model_harness" / "validate_resource_aware_frontier_policy_e2e.py").read_text(encoding="utf-8"))
OVERLAYS = []
for node in source.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "OVERLAYS" for t in node.targets):
        OVERLAYS = list(ast.literal_eval(node.value))
        break
assert OVERLAYS
if "apply_interaction_frontier_overlay.py" not in OVERLAYS:
    OVERLAYS.append("apply_interaction_frontier_overlay.py")
OVERLAYS.append("patch_intervention_menu_into_tufa.py")
OVERLAYS.append("patch_action_only_menu_enforcement_into_tufa.py")
OVERLAYS.append("patch_single_mouse_inspection_handoff_into_tufa.py")
OVERLAYS.append("patch_mouse_candidate_schema_into_tufa.py")
blob = subprocess.check_output(["git", "-C", str(BASE), "archive", "--format=zip", "33dcb90"])
with tempfile.TemporaryDirectory() as td:
    target = Path(td) / "src"
    target.mkdir()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(target)
    for name in OVERLAYS:
        subprocess.check_call([sys.executable, str(ROOT / "model_harness" / name), str(target)], cwd=ROOT, stdout=subprocess.DEVNULL)
    subprocess.check_call([sys.executable, "-m", "py_compile", str(target / "inference/agent/tool_agent.py")])
    sys.path.insert(0, str(target))
    os.environ["LOCAL_ANALYZER_MODEL_ID"] = "fake/model"
    os.environ["LOCAL_ANALYZER_BASE_URL"] = "http://127.0.0.1:1/v1"
    from inference.agent import runtime_state as rs, tool_agent as ta

    grid = [[0 for _ in range(32)] for _ in range(32)]
    for row, col in ((3,3),(3,12),(3,21),(12,3),(12,12),(12,21),(21,3),(21,12),(21,21)):
        for dr in (0, 1):
            for dc in (0, 1):
                grid[row + dr][col + dc] = 2
    frame = rs.Frame(grid=tuple(tuple(values) for values in grid), step=0, level=1)
    state = Path(td) / rs.RUNTIME_STATE_FILENAME
    rs.write_runtime_state(state, current_frame=frame, history=[])
    agent = ta.ToolAgent(model="fake/model", base_url="http://127.0.0.1:1/v1", provider="vllm")
    agent._current_valid_actions = ["MOUSE"]
    agent._ensure_session(state)
    prompt = agent._build_user_prompt(0, valid_actions=["MOUSE"], current_frame=frame, history_entries=[], previous_step_summary=None)
    assert "C1=(" in prompt and "C9=(" in prompt
    menu = list(agent._patricia_current_interventions.values())
    assert len(menu) >= 9
    first = menu[0]
    inspection = agent._run_python_tool(state, {"code": "print(current_frame.shape)", "world_model": "{}"})
    assert not inspection.step_executed
    assert agent._patricia_action_only
    assert agent._patricia_action_only_remaining == 6
    tools = agent._tools(state)
    candidate = tools[0]["function"]["parameters"]["properties"]["candidate_index"]
    assert candidate["minimum"] == 1 and candidate["maximum"] == 16
    assert tools[0]["function"]["parameters"]["required"] == ["action", "candidate_index", "world_model"]
    calls = []

    def step_env(payload):
        calls.append(payload["actions"][0])
        after = rs.Frame(grid=frame.grid, step=1, level=1)
        rs.write_runtime_state(state, current_frame=after, history=[rs.HistoryEntry(action="MOUSE", frame=after)])
        return {"executed": True, "action_num": 1, "level": 1, "reward": 0.0, "state": "NOT_FINISHED", "valid_actions": ["MOUSE"], "board_changed": False, "done": False, "level_completed": False, "game_over": False, "run_complete": False, "action_display": "MOUSE", "executed_actions": ["MOUSE"], "executed_count": 1, "transition_signature": {"cells": 0, "bbox": None, "pairs": [], "motions": []}}

    agent._step_env_callback = step_env
    result = agent._run_action_tool(state, {"action": "MOUSE", "candidate_index": 1, "world_model": "{}"})
    assert result.step_executed
    assert calls == [{"action": "MOUSE", "row": int(first["row"]), "col": int(first["col"])}]
    calls.clear()
    invalid = agent._run_action_tool(state, {"action": "MOUSE", "candidate_index": 16, "world_model": "{}"})
    assert not invalid.step_executed and not calls
    missing = agent._run_action_tool(state, {"action": "MOUSE", "world_model": "{}"})
    assert not missing.step_executed and not calls
    raw_forced = agent._run_action_tool(state, {"action": "MOUSE", "row": 31, "col": 31, "world_model": "{}"})
    assert not raw_forced.step_executed and not calls
    assert "must use candidate_index" in raw_forced.content

    rs.write_runtime_state(state, current_frame=frame, history=[])
    raw_agent = ta.ToolAgent(model="fake/model", base_url="http://127.0.0.1:1/v1", provider="vllm")
    raw_agent._current_valid_actions = ["MOUSE"]
    raw_agent._ensure_session(state)
    raw_agent._build_user_prompt(0, valid_actions=["MOUSE"], current_frame=frame, history_entries=[], previous_step_summary=None)
    raw_calls = []

    def raw_step(payload):
        raw_calls.append(payload["actions"][0])
        after = rs.Frame(grid=frame.grid, step=1, level=1)
        rs.write_runtime_state(state, current_frame=after, history=[rs.HistoryEntry(action="MOUSE", frame=after)])
        return {"executed": True, "action_num": 1, "level": 1, "reward": 0.0, "state": "NOT_FINISHED", "valid_actions": ["MOUSE"], "board_changed": False, "done": False, "level_completed": False, "game_over": False, "run_complete": False, "action_display": "MOUSE", "executed_actions": ["MOUSE"], "executed_count": 1, "transition_signature": {"cells": 0, "bbox": None, "pairs": [], "motions": []}}

    raw_agent._step_env_callback = raw_step
    raw = raw_agent._run_action_tool(state, {"action": "MOUSE", "row": 31, "col": 31, "world_model": "{}"})
    assert raw.step_executed and raw_calls == [{"action": "MOUSE", "row": 31, "col": 31}]
    print("MOUSE_CANDIDATE_SCHEMA_E2E_PASS", len(menu), (first["row"], first["col"]))
