from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_harness.interaction_frontier import summarize_targets
from paper_prize.human_frontier_coverage import _collect, _rank_bbox


def _score_state(task: tuple[str, bytes]) -> tuple[str, dict[str, Any]]:
    key, raw = task
    grid = [list(raw[start:start + 64]) for start in range(0, len(raw), 64)]
    rich = summarize_targets(grid, max_candidates=24)["candidates"]
    expanded = summarize_targets(grid, max_candidates=25)["candidates"]
    legacy = sorted(
        (item for item in expanded if item.get("legacy_rank") is not None),
        key=lambda item: tuple(item["legacy_rank"]),
    )[:24]
    return key, {"rich": rich, "legacy24": legacy}

def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    rich_hits = sum(bool(row["rich_bbox"]) for row in rows)
    legacy_hits = sum(bool(row["legacy24_bbox"]) for row in rows)
    rich_counts = [int(row["rich_count"]) for row in rows]
    legacy_counts = [int(row["legacy24_count"]) for row in rows]
    return {
        "events": n,
        "rich_bbox_n": rich_hits,
        "rich_bbox_pct": 100.0 * rich_hits / n if n else None,
        "legacy24_bbox_n": legacy_hits,
        "legacy24_bbox_pct": 100.0 * legacy_hits / n if n else None,
        "delta_points": 100.0 * (rich_hits - legacy_hits) / n if n else None,
        "rich_mean_candidates": statistics.mean(rich_counts) if rich_counts else None,
        "legacy24_mean_candidates": statistics.mean(legacy_counts) if legacy_counts else None,
    }


def analyze(data_dir: Path, workers: int) -> dict[str, Any]:
    records, states, stats = _collect(data_dir)
    scored: dict[str, dict[str, Any]] = {}
    with mp.Pool(processes=workers) as pool:
        for done, (key, value) in enumerate(pool.imap_unordered(_score_state, states.items(), chunksize=4), start=1):
            scored[key] = value
            if done % 500 == 0 or done == len(states):
                print(f"Scored {done}/{len(states)} unique states", flush=True)

    for row in records:
        item = scored[row["state_hash"]]
        target_row, target_col = int(row["row"]), int(row["col"])
        row["rich_count"] = len(item["rich"])
        row["legacy24_count"] = len(item["legacy24"])
        row["rich_bbox"] = _rank_bbox(item["rich"], target_row, target_col) is not None
        row["legacy24_bbox"] = _rank_bbox(item["legacy24"], target_row, target_col) is not None

    per_env = {}
    for env in sorted({row["env"] for row in records}):
        per_env[env] = _summarize([row for row in records if row["env"] == env])

    result = {
        **stats,
        "mouse_events": len(records),
        "unique_preclick_states": len(states),
        "all_mouse": _summarize(records),
        "winning_sessions": _summarize([row for row in records if row["won_session"]]),
        "per_env": per_env,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = analyze(args.data_dir, max(1, args.workers))
    observed = result["all_mouse"]["rich_bbox_pct"]
    if round(float(observed), 3) != 48.753:
        raise RuntimeError(f"rich24 reproduction check failed: {observed}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["all_mouse"], indent=2, sort_keys=True))
    print("WROTE", args.output)


if __name__ == "__main__":
    main()
