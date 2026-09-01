from __future__ import annotations
import ast
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = os.environ.get("PATRICIA_PAPER_POLICY", "regime").strip().lower()
if POLICY not in {"static", "v47", "regime"}:
    raise SystemExit(f"unsupported PATRICIA_PAPER_POLICY={POLICY!r}")
SOURCE = ROOT / "kaggle" / f"patricia-paper-gate-v48-{POLICY}.ipynb"
OUTPUT = ROOT / "kaggle" / f"patricia-t4-submission-v48-paper-{POLICY}.ipynb"
nb = json.loads(SOURCE.read_text(encoding="utf-8"))

# Retarget attached dataset bookkeeping; the Qwen checkpoint is a Kaggle model source.
for cell in nb["cells"]:
    text = "".join(cell.get("source", []))
    if "DATASET_SOURCES =" in text:
        start = text.index("DATASET_SOURCES =")
        end = text.index("\n", start)
        text = text[:start] + 'DATASET_SOURCES = ["jeroencottaar/taaf-kaggle-source-share", "saltb0x/arc3-vllm-wheelhouse-v0271-cu129"]' + text[end:]
        cell["source"] = text.splitlines(keepends=True)
        break
else:
    raise RuntimeError("DATASET_SOURCES cell not found")
# Transform the proven Tufa setup command instead of duplicating its integration plumbing.
for cell in nb["cells"]:
    text = "".join(cell.get("source", []))
    if "# Solver setup commands" not in text:
        continue
    old_loop = '''# Solver setup commands (wheels, vLLM server startup, ...) run before the benchmark loads.
env = _command_env()
for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):
    print(f"taaf.kaggle: setup command: {command}", flush=True)
    subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)
    # Re-read in case the command persisted new env keys.
    env = _command_env()
    os.environ.update(env)
'''
    new_loop = '''# Patricia T4/AWQ serving transform: preserve Tufa's integration, replace only runtime/model/hardware.
os.environ["KAGGLE_GPU_TYPE"] = "t4"
os.environ["KAGGLE_GPU_COUNT"] = "2"
os.environ["LOCAL_ANALYZER_MAX_OUTPUT"] = "512"
env = _command_env()
for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):
    command = command.replace("driessmit1'\\nWHEELHOUSE_SLUG = 'arc3-vllm-h100-wheelhouse-v3", "saltb0x'\\nWHEELHOUSE_SLUG = 'arc3-vllm-wheelhouse-v0271-cu129")
    command = command.replace("MODEL_OWNER = 'driessmit1'", "MODEL_OWNER = 'bachhg'")
    command = command.replace("MODEL_SLUG = 'vrfai-qwen3-6-27b-fp8-hf-snapshot'", "MODEL_SLUG = 'qwen3-6-27b-awq'")
    command = command.replace("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'", "SERVED_MODEL_NAME = 'Qwen3.6-27B-AWQ'")
'''
    if old_loop not in text:
        raise RuntimeError("setup loop anchor missing")
    text = text.replace(old_loop, new_loop, 1)
    cell["source"] = text.splitlines(keepends=True)
    break
else:
    raise RuntimeError("setup cell not found")
# Expand the transform with the serving flags that were proven by the multimodal smoke test.
for cell in nb["cells"]:
    text = "".join(cell.get("source", []))
    needle = '    command = command.replace("SERVED_MODEL_NAME = \'vrfai/Qwen3.6-27B-FP8\'", "SERVED_MODEL_NAME = \'Qwen3.6-27B-AWQ\'")\n'
    if needle not in text:
        continue
    extra = needle + '''    command = command.replace("VLLM_MAX_MODEL_LEN = 65536", "VLLM_MAX_MODEL_LEN = 8192")
    command = command.replace("ANALYZER_CONTEXT_WINDOW = 32768", "ANALYZER_CONTEXT_WINDOW = 8192")
    command = command.replace("'LOCAL_ANALYZER_YIELD_SECONDS': '60'", "'LOCAL_ANALYZER_YIELD_SECONDS': '0'")
    command = command.replace("VLLM_TENSOR_PARALLEL_SIZE = 1", "VLLM_TENSOR_PARALLEL_SIZE = 2")
    command = command.replace("STAMP_TEXT = 'vllm==0.19.0 torch==2.10.0 flashinfer==0.6.6\\\\n'", "STAMP_TEXT = 'vllm==0.27.1 torch==2.13.0 compressed-tensors==0.17.0\\\\n'")
    command = command.replace("MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)", "MODEL_PATH = next((p.parent for p in Path('/kaggle/input/models').rglob('config.json') if 'qwen3-6-27b-awq' in str(p).lower()), None)\\nif MODEL_PATH is None: raise FileNotFoundError('Qwen3.6 AWQ model source not found')")
    command = command.replace("            'TRANSFORMERS_NO_TORCHVISION': '1',\\n", "")
    command = command.replace("            'VLLM_NO_USAGE_STATS': '1',", "            'VLLM_NO_USAGE_STATS': '1',\\n            'VLLM_LOGGING_LEVEL': 'WARNING',\\n            'VLLM_LOGGING_COLOR': '0',\\n            'VLLM_USE_FLASHINFER_SAMPLER': '0',\\n            'PYTHONUTF8': '1',\\n            'PYTHONIOENCODING': 'utf-8',\\n            'LANG': 'C.UTF-8',\\n            'LC_ALL': 'C.UTF-8',")
'''
    text = text.replace(needle, extra, 1)
    cell["source"] = text.splitlines(keepends=True)
    break
else:
    raise RuntimeError("T4 transform insertion point missing")
# Add eager/T4 memory flags and restore execution of the transformed setup command.
for cell in nb["cells"]:
    text = "".join(cell.get("source", []))
    anchor = '    command = command.replace("            \'VLLM_NO_USAGE_STATS\': \'1\',", "            \'VLLM_NO_USAGE_STATS\': \'1\',\\n            \'VLLM_LOGGING_LEVEL\': \'WARNING\',\\n            \'VLLM_LOGGING_COLOR\': \'0\',\\n            \'VLLM_USE_FLASHINFER_SAMPLER\': \'0\',\\n            \'PYTHONUTF8\': \'1\',\\n            \'PYTHONIOENCODING\': \'utf-8\',\\n            \'LANG\': \'C.UTF-8\',\\n            \'LC_ALL\': \'C.UTF-8\',")\n'
    if anchor not in text:
        continue
    addition = anchor + '''    command = command.replace("        str(VLLM_MAX_MODEL_LEN),\\n    ]", "        str(VLLM_MAX_MODEL_LEN),\\n        '--enforce-eager',\\n        '--dtype', 'half',\\n        '--gpu-memory-utilization', '0.90',\\n        '--max-num-seqs', '1',\\n        '--max-num-batched-tokens', '8192',\\n    ]")
    command = command.replace("    'MULTIMODAL_UPSCALE': '4',", "    'MULTIMODAL_UPSCALE': '4',\\n    'MULTIMODAL_STYLE': 'outline',")
    print("taaf.kaggle: transformed T4 setup command", flush=True)
    subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)
    env = _command_env()
    os.environ.update(env)

# The setup runs in a subprocess, so explicitly put its wheelhouse target first
# in this notebook process before importing arcengine/Pillow or the agent.
vllm_site = "/kaggle/working/vllm-site-packages"
if os.path.isdir(vllm_site):
    while vllm_site in sys.path:
        sys.path.remove(vllm_site)
    sys.path.insert(0, vllm_site)
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = vllm_site + (os.pathsep + current_pythonpath if current_pythonpath else "")
    print("taaf.kaggle: prepended runtime site-packages:", vllm_site, flush=True)
    # Kaggle may have imported its system Pillow earlier; sys.path alone cannot replace cached modules.
    for _name in [n for n in list(sys.modules) if n == "PIL" or n.startswith("PIL.")]:
        del sys.modules[_name]
    import PIL as _pil_runtime
    _pil_path = str(Path(_pil_runtime.__file__).resolve())
    print("taaf.kaggle: Pillow runtime:", _pil_path, getattr(_pil_runtime, "__version__", "unknown"), flush=True)
    if not _pil_path.startswith(vllm_site):
        raise RuntimeError(f"Pillow did not reload from wheelhouse: {_pil_path}")
'''
    text = text.replace(anchor, addition, 1)
    cell["source"] = text.splitlines(keepends=True)
    break
else:
    raise RuntimeError("runtime flag insertion point missing")
# Cheap end-to-end integration check before the five-game score experiment.
for cell in nb["cells"]:
    text = "".join(cell.get("source", []))
    if "PATRICIA_GATE_GAMES" in text and "max_actions_per_game" in text:
        text = text.replace("bm.solver.max_actions_per_game = 300", "bm.solver.max_actions_per_game = 60")
        text = text.replace(f"patricia-paper-gate-v48-{POLICY}", f"patricia-t4-submission-v48-paper-{POLICY}")
        text = text.replace("ls20,sp80,r11l,m0r0,ft09", "r11l")
        text = text.replace("# Patricia five-game reasoning gate. True competition reruns remain unrestricted.", "# Patricia submission smoke gate. True competition reruns remain unrestricted.")
        governor_anchor = "    bm.solver.save_request_logs = True\n"
        governor = governor_anchor + (
            "bm.solver.concurrency = 1\n"
            "bm.solver.analyzer_timeout = 180.0\n"
            "bm.solver.max_runtime_s_per_game = 240.0\n"
            "print(\"Patricia serving governor: concurrency=1 analyzer_timeout=180s max_game=240s max_output=512\")\n"
        )
        if governor_anchor not in text:
            raise RuntimeError("serving governor anchor missing")
        text = text.replace(governor_anchor, governor, 1)
        cell["source"] = text.splitlines(keepends=True)
        break
else:
    raise RuntimeError("gate settings cell not found")

# Dev kernels already carry the 25 public GameAPI objects in benchmark_initial.pkl.
# Filter those hashed env_names by their stable short game prefix instead of
# reconstructing environments from a competition directory Kaggle does not mount.
for cell in nb["cells"]:
    text = "".join(cell.get("source", []))
    old = """else:
    # Interactive run: play the bundled competition environments offline (no gateway).
    # The competition's environment files ship alongside the wheelhouse in the competition dataset.
    competition_env_files = str(Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels").parent / "environment_files")
    bm.games = _offline_games(competition_env_files)
"""
    if old not in text:
        continue
    new = """else:
    gate_games = {x.strip() for x in os.environ.get("PATRICIA_GATE_GAMES", "").split(",") if x.strip()}
    if gate_games:
        bm.games = [g for g in bm.games if str(getattr(g, "env_name", "")).split("-", 1)[0] in gate_games]
    if not bm.games:
        raise RuntimeError(f"No benchmark games matched {sorted(gate_games)}")
    # Resolve the official competition environment mount dynamically; Kaggle mount roots vary.
    import dataclasses
    _env_probe = next(Path('/kaggle/input').rglob('ls20.py'), None)
    if _env_probe is None:
        raise RuntimeError('Could not locate official competition environment_files under /kaggle/input')
    _env_root = str(_env_probe.parents[2])
    for _game in bm.games:
        _game.arcade_spec = dataclasses.replace(_game.arcade_spec, environments_dir=_env_root)
    print("taaf.kaggle: selected existing benchmark games:", [g.env_name for g in bm.games], flush=True)
    print("taaf.kaggle: explicit environment root:", _env_root, flush=True)
"""
    text = text.replace(old, new, 1)
    cell["source"] = text.splitlines(keepends=True)
    break
else:
    raise RuntimeError("offline game-selection block not found")

# Give the integration experiment 60 minutes after expensive model setup, not after notebook launch.
for cell in nb["cells"]:
    text = "".join(cell.get("source", []))
    old_timer = "soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=900)"
    if old_timer in text:
        text = text.replace(old_timer, "soft_end = datetime.now() + timedelta(seconds=1200)", 1)
        cell["source"] = text.splitlines(keepends=True)
        break
else:
    raise RuntimeError("integration soft-end timer not found")

# Shared T4 compatibility overlay: allow one inspection-only call, then force a
# transparent rotating valid probe if this quantized model still refuses action(...).
# This cell is byte-identical in Patricia and pristine-control notebooks.
for cell in nb["cells"]:
    text = "".join(cell.get("source", []))
    if "Pillow did not reload from wheelhouse" not in text:
        continue
    anti_stall_runtime = r'''
# T4/AWQ anti-stall compatibility patch, applied identically to both A/B arms.
_tool_agent_path = PATRICIA_HARNESS_REPO / "inference" / "agent" / "tool_agent.py"
_tool_agent_src = _tool_agent_path.read_text(encoding="utf-8")
_anti_stall_anchor = '        valid_actions = list(_normalize_valid_actions(self._current_valid_actions))\n'
_anti_stall_insert = (
    '        _anti_stall_has_action = bool(re.search(r"(?<!\\w)action\\s*\\(", code))\n'
    '        if _anti_stall_has_action:\n'
    '            self._anti_stall_inspection_streak = 0\n'
    '        else:\n'
    '            _anti_stall_streak = int(getattr(self, "_anti_stall_inspection_streak", 0)) + 1\n'
    '            self._anti_stall_inspection_streak = _anti_stall_streak\n'
    '            if _anti_stall_streak >= 2 and valid_actions:\n'
    '                _anti_stall_index = int(getattr(self, "_anti_stall_probe_index", 0))\n'
    '                _anti_stall_probe = valid_actions[_anti_stall_index % len(valid_actions)]\n'
    '                self._anti_stall_probe_index = _anti_stall_index + 1\n'
    '                self._anti_stall_inspection_streak = 0\n'
    '                code += (\n'
    '                    "\\n# T4 anti-stall compatibility probe after one inspection-only call.\\n"\n'
    '                    f"# T4_ANTI_STALL_FORCED_ACTION={_anti_stall_probe}\\n"\n'
    '                    f"print(\\"ANTI_STALL_PROBE: {_anti_stall_probe}\\")\\n"\n'
    '                    f"action([{_anti_stall_probe!r}])\\n"\n'
    '                )\n'
    '                try:\n'
    '                    compile(code, "<python_tool_anti_stall>", "exec")\n'
    '                except SyntaxError as exc:\n'
    '                    return _ToolDispatchResult(json.dumps({"error": f"Anti-stall probe injection failed: {exc}"}, indent=2))\n'
)
if _anti_stall_anchor not in _tool_agent_src:
    raise RuntimeError("T4 anti-stall anchor missing from tool_agent.py")
if '_anti_stall_probe = valid_actions[_anti_stall_index % len(valid_actions)]' not in _tool_agent_src:
    _tool_agent_src = _tool_agent_src.replace(_anti_stall_anchor, _anti_stall_anchor + _anti_stall_insert, 1)
    _tool_agent_path.write_text(_tool_agent_src, encoding="utf-8")
print("taaf.kaggle: T4 anti-stall compatibility patch ready", flush=True)
'''
    text += anti_stall_runtime
    cell["source"] = text.splitlines(keepends=True)
    break
else:
    raise RuntimeError("runtime cell for T4 anti-stall patch not found")

nb["cells"][0]["source"] = [f"# Patricia ARC-AGI-3 T4 Paper Prize representative submission v48 policy={POLICY}\n", "\n", "Representative full competition rerun for the final Paper Prize interaction-policy package.\n"]
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
for _cell_index, _cell in enumerate(nb["cells"]):
    if _cell.get("cell_type") != "code":
        continue
    _source = "".join(_cell.get("source", []))
    compile(_source, f"generated-cell-{_cell_index}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)

OUTPUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"built {OUTPUT} bytes={OUTPUT.stat().st_size} cells={len(nb['cells'])}")
