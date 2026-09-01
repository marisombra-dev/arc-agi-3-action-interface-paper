from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_harness.interaction_frontier import summarize_targets

GRID_SIDE = 64
ACTION_SPACE = GRID_SIDE * GRID_SIDE


def _grid_bytes(frame_layers: list[Any]) -> bytes:
    return np.asarray(frame_layers[-1], dtype=np.uint8).reshape(-1).tobytes()


def _state_key(raw: bytes) -> str:
    return hashlib.blake2s(raw, digest_size=8).hexdigest()


def _score_state(task: tuple[str, bytes]) -> tuple[str, list[dict[str, Any]]]:
    key, raw = task
    grid = [list(raw[start : start + GRID_SIDE]) for start in range(0, len(raw), GRID_SIDE)]
    candidates = summarize_targets(grid, max_candidates=24)["candidates"]
    compact = [
        {
            "row": int(item["row"]),
            "col": int(item["col"]),
            "bbox": [int(value) for value in item.get("bbox", [])],
            "roles": list(item.get("roles", [])),
            "legacy_eligible": bool(item.get("legacy_eligible")),
        }
        for item in candidates
    ]
    return key, compact


def _rank_exact(candidates: list[dict[str, Any]], row: int, col: int) -> int | None:
    for idx, item in enumerate(candidates, start=1):
        if item["row"] == row and item["col"] == col:
            return idx
    return None


def _rank_near(candidates: list[dict[str, Any]], row: int, col: int, radius: int = 1) -> int | None:
    for idx, item in enumerate(candidates, start=1):
        if abs(item["row"] - row) <= radius and abs(item["col"] - col) <= radius:
            return idx
    return None


def _rank_bbox(candidates: list[dict[str, Any]], row: int, col: int) -> int | None:
    for idx, item in enumerate(candidates, start=1):
        bbox = item["bbox"]
        if len(bbox) == 4 and bbox[0] <= row <= bbox[2] and bbox[1] <= col <= bbox[3]:
            return idx
    return None


def _pct(numerator: int, denominator: int) -> float | None:
    return round(100.0 * numerator / denominator, 3) if denominator else None


def _subset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    counts = [int(row["candidate_count"]) for row in rows]
    out: dict[str, Any] = {
        "events": total,
        "mean_candidate_count": round(statistics.mean(counts), 3) if counts else None,
        "median_candidate_count": statistics.median(counts) if counts else None,
        "mean_action_space_compression_pct": (
            round(100.0 * (1.0 - statistics.mean(counts) / ACTION_SPACE), 4) if counts else None
        ),
    }
    for key in ("exact_rank", "near1_rank", "bbox_rank"):
        for cutoff in (8, 24):
            n = sum(row[key] is not None and int(row[key]) <= cutoff for row in rows)
            out[f"{key}_top{cutoff}_n"] = n
            out[f"{key}_top{cutoff}_pct"] = _pct(n, total)
    for key in ("legacy_exact", "legacy_bbox"):
        n = sum(bool(row[key]) for row in rows)
        out[f"{key}_n"] = n
        out[f"{key}_pct"] = _pct(n, total)
    return out


def _collect(data_dir: Path) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, int]]:
    records: list[dict[str, Any]] = []
    states: dict[str, bytes] = {}
    stats = {"sessions": 0, "parsed_events": 0, "skipped_mouse_events": 0}
    shards = sorted(data_dir.glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No train-*.parquet shards in {data_dir}")

    for shard_index, shard in enumerate(shards, start=1):
        frame = pd.read_parquet(shard, columns=["env", "guid", "trajectory", "won"])
        for _, session in frame.iterrows():
            stats["sessions"] += 1
            trajectory = json.loads(session["trajectory"])
            stats["parsed_events"] += len(trajectory)
            for event_index in range(1, len(trajectory)):
                current = trajectory[event_index]["data"]
                action = current.get("action_input") or {}
                if str(action.get("id")) != "6":
                    continue
                previous = trajectory[event_index - 1]["data"]
                prev_frames = previous.get("frame") or []
                click = action.get("data") or {}
                if not prev_frames or click.get("x") is None or click.get("y") is None:
                    stats["skipped_mouse_events"] += 1
                    continue
                raw = _grid_bytes(prev_frames)
                key = _state_key(raw)
                states.setdefault(key, raw)
                current_frames = current.get("frame") or []
                current_raw = _grid_bytes(current_frames) if current_frames else b""
                changed = sum(a != b for a, b in zip(raw, current_raw)) if len(raw) == len(current_raw) else None
                level_before = int(previous.get("levels_completed") or 0)
                level_after = int(current.get("levels_completed") or 0)
                objective = level_after > level_before or str(current.get("state")) == "WIN"
                records.append(
                    {
                        "env": str(session["env"]),
                        "guid": str(session["guid"]),
                        "won_session": bool(int(session["won"])),
                        "event_index": event_index,
                        "level_before": level_before,
                        "level_after": level_after,
                        "row": int(click["y"]),
                        "col": int(click["x"]),
                        "state_hash": key,
                        "changed_cells": changed,
                        "effectful": bool((changed or 0) > 0 or objective),
                        "objective_progress": bool(objective),
                    }
                )
        print(
            f"Collected shard {shard_index}/{len(shards)}: sessions={stats['sessions']} "
            f"mouse={len(records)} unique_states={len(states)}",
            flush=True,
        )
    return records, states, stats


def analyze(data_dir: Path, workers: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    records, states, stats = _collect(data_dir)
    scored: dict[str, list[dict[str, Any]]] = {}
    tasks = list(states.items())
    print(f"Scoring {len(tasks)} unique pre-click states with {workers} workers", flush=True)
    with mp.Pool(processes=workers) as pool:
        for done, (key, candidates) in enumerate(pool.imap_unordered(_score_state, tasks, chunksize=4), start=1):
            scored[key] = candidates
            if done % 250 == 0 or done == len(tasks):
                print(f"Scored {done}/{len(tasks)} unique states", flush=True)

    for record in records:
        candidates = scored[record["state_hash"]]
        row, col = int(record["row"]), int(record["col"])
        record["candidate_count"] = len(candidates)
        record["exact_rank"] = _rank_exact(candidates, row, col)
        record["near1_rank"] = _rank_near(candidates, row, col)
        record["bbox_rank"] = _rank_bbox(candidates, row, col)
        legacy = [item for item in candidates if item["legacy_eligible"]]
        record["legacy_exact"] = _rank_exact(legacy, row, col) is not None
        record["legacy_bbox"] = _rank_bbox(legacy, row, col) is not None

    events = pd.DataFrame(records)
    summary: dict[str, Any] = {
        **stats,
        "mouse_events": len(records),
        "unique_preclick_states": len(states),
        "all_mouse": _subset_summary(records),
        "winning_sessions": _subset_summary([row for row in records if row["won_session"]]),
        "effectful_mouse": _subset_summary([row for row in records if row["effectful"]]),
        "objective_mouse": _subset_summary([row for row in records if row["objective_progress"]]),
    }
    per_env: dict[str, Any] = {}
    for env in sorted({row["env"] for row in records}):
        env_rows = [row for row in records if row["env"] == env]
        per_env[env] = _subset_summary(env_rows)
        per_env[env]["winning_events"] = sum(row["won_session"] for row in env_rows)
        per_env[env]["effectful_events"] = sum(row["effectful"] for row in env_rows)
    summary["per_env"] = per_env
    return events, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("paper_prize/results"))
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events, summary = analyze(args.data_dir, max(1, args.workers))
    events.to_csv(args.output_dir / "human_frontier_events.csv", index=False)
    (args.output_dir / "human_frontier_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    headline = {key: value for key, value in summary.items() if key != "per_env"}
    print(json.dumps(headline, indent=2), flush=True)
    print(f"Wrote {len(events)} mouse-event rows to {args.output_dir}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
