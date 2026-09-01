from __future__ import annotations

import argparse
from collections import Counter
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

from model_harness.interaction_frontier import (
    enumerate_interventions,
    observe_intervention,
    rank_interventions,
)

GRID_SIDE = 64


def _grid_bytes(frame_layers: list[Any]) -> bytes:
    return np.asarray(frame_layers[-1], dtype=np.uint8).reshape(-1).tobytes()


def _key(raw: bytes) -> str:
    return hashlib.blake2s(raw, digest_size=8).hexdigest()


def _grid(raw: bytes) -> list[list[int]]:
    return [list(raw[start : start + GRID_SIDE]) for start in range(0, len(raw), GRID_SIDE)]


def _frontier_task(task: tuple[str, bytes]) -> tuple[str, dict[str, Any]]:
    key, raw = task
    frontier = enumerate_interventions(_grid(raw), ["MOUSE"], max_targets=16)
    compact = {
        "state_hash": frontier.get("state_hash"),
        "interior_hash": frontier.get("interior_hash"),
        "interventions": [],
    }
    for item in frontier.get("interventions", []):
        compact["interventions"].append(
            {
                "id": item.get("id"),
                "action": "MOUSE",
                "row": int(item["row"]),
                "col": int(item["col"]),
                "target_id": item.get("target_id"),
                "target_class": item.get("target_class"),
                "target_roles": list(item.get("target_roles", [])),
                "target_bbox": list(item.get("target_bbox") or []),
                "target_size": item.get("target_size"),
                "target_copies": item.get("target_copies"),
                "intervention_class": item.get("intervention_class"),
                "priority": int(item.get("priority") or 0),
                "target_rank": item.get("target_rank"),
            }
        )
    return key, compact


def _bbox_contains(item: dict[str, Any], row: int, col: int) -> bool:
    bbox = item.get("target_bbox") or []
    return len(bbox) == 4 and bbox[0] <= row <= bbox[2] and bbox[1] <= col <= bbox[3]


def _match_rank(items: list[dict[str, Any]], row: int, col: int) -> int | None:
    for index, item in enumerate(items, start=1):
        if _bbox_contains(item, row, col):
            return index
    return None


def _representative(items: list[dict[str, Any]], row: int, col: int) -> dict[str, Any]:
    for item in items:
        if _bbox_contains(item, row, col):
            return dict(item)
    return {
        "action": "MOUSE",
        "row": row,
        "col": col,
        "target_id": None,
        "target_class": "unmodeled_human_click",
        "target_roles": [],
        "target_bbox": [row, col, row, col],
        "intervention_class": "control:MOUSE->unmodeled_human_click",
        "priority": -999,
    }


def _pct(count: int, total: int) -> float | None:
    return round(100.0 * count / total, 3) if total else None


def _collect(data_dir: Path) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, int]]:
    rows: list[dict[str, Any]] = []
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
                click = action.get("data") or {}
                before_layers = previous.get("frame") or []
                after_layers = current.get("frame") or []
                if not before_layers or not after_layers or click.get("x") is None or click.get("y") is None:
                    stats["skipped_mouse_events"] += 1
                    continue
                before_raw, after_raw = _grid_bytes(before_layers), _grid_bytes(after_layers)
                before_key, after_key = _key(before_raw), _key(after_raw)
                states.setdefault(before_key, before_raw)
                states.setdefault(after_key, after_raw)
                level_before = int(previous.get("levels_completed") or 0)
                level_after = int(current.get("levels_completed") or 0)
                rows.append(
                    {
                        "env": str(session["env"]),
                        "guid": str(session["guid"]),
                        "won_session": bool(int(session["won"])),
                        "event_index": event_index,
                        "level_before": level_before,
                        "level_after": level_after,
                        "terminal": str(current.get("state")) == "WIN",
                        "row": int(click["y"]),
                        "col": int(click["x"]),
                        "before_key": before_key,
                        "after_key": after_key,
                    }
                )
        print(f"Collected {shard_index}/{len(shards)} sessions={stats['sessions']} mouse={len(rows)} states={len(states)}", flush=True)
    return rows, states, stats



def _observe_task(task: tuple[tuple[str, str, int], bytes, bytes, dict[str, Any], bool, bool]) -> tuple[tuple[str, str, int], dict[str, Any]]:
    event_id, before_raw, after_raw, representative, level_completed, terminal = task
    observation = observe_intervention(
        _grid(before_raw),
        _grid(after_raw),
        representative,
        level_completed=level_completed,
        terminal=terminal,
    )
    return event_id, observation


def _rank_variant(
    frontier: dict[str, Any],
    observations: list[dict[str, Any]],
    pending_source: dict[str, Any] | None,
    *,
    unseen_bonus: int = 0,
    changed_bonus: int = 0,
    objective_bonus: int = 0,
    nochange_penalty: int = 0,
    exact_penalty: int = 0,
    interior_penalty: int = 0,
    class_repeat_penalty: int = 0,
    class_repeat_cap: int = 0,
    pending_bonus: int = 0,
) -> list[dict[str, Any]]:
    state_hash = str(frontier.get("state_hash") or "")
    interior_hash = str(frontier.get("interior_hash") or "")
    exact_attempts: Counter[tuple[str, str]] = Counter()
    interior_attempts: Counter[tuple[str, str]] = Counter()
    class_attempts: Counter[str] = Counter()
    changed_classes: set[str] = set()
    no_change_classes: Counter[str] = Counter()
    objective_classes: Counter[str] = Counter()
    for observation in observations:
        intervention_class = str(observation.get("intervention_class") or "")
        key = str(observation.get("intervention_key") or observation.get("target_id") or "")
        class_attempts[intervention_class] += 1
        if str(observation.get("before_hash") or "") == state_hash:
            exact_attempts[(intervention_class, key)] += 1
        if interior_hash and str(observation.get("before_interior_hash") or "") == interior_hash:
            interior_attempts[(intervention_class, key)] += 1
        if observation.get("level_completed") or observation.get("terminal") or observation.get("reward"):
            objective_classes[intervention_class] += 1
        elif int(observation.get("changed_cells") or 0) > 0:
            changed_classes.add(intervention_class)
        else:
            no_change_classes[intervention_class] += 1

    pending_key = str((pending_source or {}).get("intervention_key") or (pending_source or {}).get("target_id") or "")
    ranked: list[dict[str, Any]] = []
    for intervention in frontier.get("interventions", []):
        item = dict(intervention)
        intervention_class = str(item.get("intervention_class") or "")
        key = f"MOUSE:{int(item.get('row') or 0)}:{int(item.get('col') or 0)}"
        score = int(item.get("priority") or 0)
        if class_attempts[intervention_class] == 0:
            score += unseen_bonus
        if intervention_class in changed_classes:
            score += changed_bonus
        score += objective_bonus * objective_classes[intervention_class]
        score -= nochange_penalty * no_change_classes[intervention_class]
        score -= exact_penalty * exact_attempts[(intervention_class, key)]
        score -= interior_penalty * interior_attempts[(intervention_class, key)]
        if class_repeat_penalty:
            repeat_cost = class_repeat_penalty * max(0, class_attempts[intervention_class] - 1)
            score -= min(class_repeat_cap, repeat_cost) if class_repeat_cap else repeat_cost
        if pending_source and item.get("action") == "MOUSE" and key != pending_key:
            score += pending_bonus
        item["intervention_key"] = key
        item["frontier_score"] = score
        ranked.append(item)
    return sorted(
        ranked,
        key=lambda item: (
            -int(item["frontier_score"]),
            str(item.get("action") or ""),
            str(item.get("intervention_key") or ""),
        ),
    )


ABLATION_POLICIES = {
    "outcome_only": dict(changed_bonus=6, objective_bonus=120, nochange_penalty=12),
    "revisit_only": dict(exact_penalty=90, interior_penalty=55),
    "pending_only": dict(pending_bonus=10),
    "no_novelty": dict(changed_bonus=6, objective_bonus=120, nochange_penalty=12, exact_penalty=90, interior_penalty=55, pending_bonus=10),
    "no_revisit": dict(unseen_bonus=30, changed_bonus=6, objective_bonus=120, nochange_penalty=12, class_repeat_penalty=3, class_repeat_cap=24, pending_bonus=10),
    "no_outcome": dict(unseen_bonus=30, exact_penalty=90, interior_penalty=55, class_repeat_penalty=3, class_repeat_cap=24, pending_bonus=10),
    "no_pending": dict(unseen_bonus=30, changed_bonus=6, objective_bonus=120, nochange_penalty=12, exact_penalty=90, interior_penalty=55, class_repeat_penalty=3, class_repeat_cap=24),
}


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    covered = [r for r in rows if r["static_rank"] is not None]
    out: dict[str, Any] = {
        "events": total,
        "covered_events": len(covered),
        "covered_pct": _pct(len(covered), total),
    }
    for prefix in ("static", "causal"):
        for cutoff in (8, 16, 24):
            n = sum(r[f"{prefix}_rank"] is not None and int(r[f"{prefix}_rank"]) <= cutoff for r in rows)
            out[f"{prefix}_top{cutoff}_n"] = n
            out[f"{prefix}_top{cutoff}_pct"] = _pct(n, total)
            conditional = sum(
                r[f"{prefix}_rank"] is not None and int(r[f"{prefix}_rank"]) <= cutoff for r in covered
            )
            out[f"{prefix}_top{cutoff}_given_covered_pct"] = _pct(conditional, len(covered))
    moved_up = [r for r in covered if r["causal_rank"] < r["static_rank"]]
    moved_down = [r for r in covered if r["causal_rank"] > r["static_rank"]]
    unchanged = [r for r in covered if r["causal_rank"] == r["static_rank"]]
    out["moved_up_pct_given_covered"] = _pct(len(moved_up), len(covered))
    out["moved_down_pct_given_covered"] = _pct(len(moved_down), len(covered))
    out["unchanged_pct_given_covered"] = _pct(len(unchanged), len(covered))
    deltas = [int(r["static_rank"]) - int(r["causal_rank"]) for r in covered]
    out["mean_rank_improvement_given_covered"] = round(statistics.mean(deltas), 3) if deltas else None
    return out


def _ablation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    covered = [row for row in rows if row.get("static_rank") is not None]
    total = len(rows)
    out: dict[str, Any] = {}
    for name in ["causal", *ABLATION_POLICIES.keys()]:
        key = f"{name}_rank"
        top8 = sum(row.get(key) is not None and int(row[key]) <= 8 for row in rows)
        top8_covered = sum(row.get(key) is not None and int(row[key]) <= 8 for row in covered)
        deltas = [int(row["static_rank"]) - int(row[key]) for row in covered if row.get(key) is not None]
        moved_up = sum(row.get(key) is not None and int(row[key]) < int(row["static_rank"]) for row in covered)
        moved_down = sum(row.get(key) is not None and int(row[key]) > int(row["static_rank"]) for row in covered)
        out[name] = {
            "top8_pct": _pct(top8, total),
            "top8_given_covered_pct": _pct(top8_covered, len(covered)),
            "mean_rank_improvement_given_covered": round(statistics.mean(deltas), 3) if deltas else None,
            "moved_up_pct_given_covered": _pct(moved_up, len(covered)),
            "moved_down_pct_given_covered": _pct(moved_down, len(covered)),
        }
    return out


def analyze(data_dir: Path, workers: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    records, states, stats = _collect(data_dir)
    frontiers: dict[str, dict[str, Any]] = {}
    tasks = list(states.items())
    print(f"Scoring {len(tasks)} unique states with {workers} workers", flush=True)
    with mp.Pool(processes=workers) as pool:
        for done, (key, frontier) in enumerate(pool.imap_unordered(_frontier_task, tasks, chunksize=4), start=1):
            frontiers[key] = frontier
            if done % 500 == 0 or done == len(tasks):
                print(f"Scored {done}/{len(tasks)} states", flush=True)

    def observation_tasks():
        for record in records:
            static = list(frontiers[record["before_key"]]["interventions"])
            representative = _representative(static, int(record["row"]), int(record["col"]))
            event_id = (str(record["env"]), str(record["guid"]), int(record["event_index"]))
            yield (
                event_id,
                states[record["before_key"]],
                states[record["after_key"]],
                representative,
                int(record["level_after"]) > int(record["level_before"]),
                bool(record["terminal"]),
            )

    observations_by_event: dict[tuple[str, str, int], dict[str, Any]] = {}
    print(f"Classifying {len(records)} MOUSE transition effects with {workers} workers", flush=True)
    with mp.Pool(processes=workers) as pool:
        for done, (event_id, observation) in enumerate(
            pool.imap_unordered(_observe_task, observation_tasks(), chunksize=8), start=1
        ):
            observations_by_event[event_id] = observation
            if done % 1000 == 0 or done == len(records):
                print(f"Classified {done}/{len(records)} MOUSE effects", flush=True)

    by_session: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        by_session.setdefault((record["env"], record["guid"]), []).append(record)

    dummy_nonmouse = {
        "intervention_class": "control:NON_MOUSE",
        "intervention_key": "NON_MOUSE",
        "before_hash": "",
        "before_interior_hash": "",
        "changed_cells": 0,
        "level_completed": False,
        "terminal": False,
        "reward": 0.0,
    }
    output: list[dict[str, Any]] = []
    replayed = 0
    for session_rows in by_session.values():
        session_rows.sort(key=lambda row: int(row["event_index"]))
        observations: list[dict[str, Any]] = []
        pending_source: dict[str, Any] | None = None
        prior_event_index = 0
        for record in session_rows:
            event_index = int(record["event_index"])
            gap = max(0, event_index - prior_event_index - 1)
            if gap:
                pending_source = None
                if gap >= 64:
                    observations = [dummy_nonmouse] * 64
                else:
                    observations = (observations + [dummy_nonmouse] * gap)[-64:]

            frontier = frontiers[record["before_key"]]
            static = list(frontier["interventions"])
            causal = rank_interventions(frontier, observations, pending_source=pending_source)
            shadow_v47 = _rank_variant(
                frontier,
                observations,
                pending_source,
                unseen_bonus=30,
                changed_bonus=6,
                objective_bonus=120,
                nochange_penalty=12,
                exact_penalty=90,
                interior_penalty=55,
                class_repeat_penalty=3,
                class_repeat_cap=24,
                pending_bonus=10,
            )
            production_keys = [item.get("intervention_key") for item in causal]
            shadow_keys = [item.get("intervention_key") for item in shadow_v47]
            if production_keys != shadow_keys:
                raise AssertionError("v47 shadow ranking diverged from production rank_interventions")
            ablation_rankings = {
                name: _rank_variant(frontier, observations, pending_source, **weights)
                for name, weights in ABLATION_POLICIES.items()
            }
            row, col = int(record["row"]), int(record["col"])
            event_id = (str(record["env"]), str(record["guid"]), event_index)
            record = dict(record)
            record["history_count"] = len(observations)
            record["mouse_history_count"] = sum(
                str(item.get("intervention_class") or "").startswith("control:MOUSE") for item in observations
            )
            record["static_rank"] = _match_rank(static, row, col)
            record["causal_rank"] = _match_rank(causal, row, col)
            for name, ranking in ablation_rankings.items():
                record[f"{name}_rank"] = _match_rank(ranking, row, col)
            output.append(record)

            obs = observations_by_event[event_id]
            observations.append(obs)
            observations = observations[-64:]
            level_completed = int(record["level_after"]) > int(record["level_before"])
            pending_source = (
                obs
                if obs.get("followup_warranted") and not level_completed and not bool(record["terminal"])
                else None
            )
            prior_event_index = event_index
            replayed += 1
            if replayed % 2000 == 0 or replayed == len(records):
                print(f"Replayed {replayed}/{len(records)} MOUSE decisions", flush=True)

    summary: dict[str, Any] = {**stats, "mouse_events": len(output), "unique_states": len(states)}
    summary["all_mouse"] = _summarize(output)
    summary["winning_sessions"] = _summarize([row for row in output if row["won_session"]])
    summary["with_mouse_history"] = _summarize(
        [row for row in output if int(row["mouse_history_count"]) > 0]
    )
    summary["ablation_all_mouse"] = _ablation_summary(output)
    summary["ablation_with_mouse_history"] = _ablation_summary(
        [row for row in output if int(row["mouse_history_count"]) > 0]
    )
    summary["per_env"] = {}
    for env in sorted({row["env"] for row in output}):
        env_rows = [row for row in output if row["env"] == env]
        summary["per_env"][env] = _summarize(env_rows)
    return pd.DataFrame(output), summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("paper_prize/causal_results"))
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events, summary = analyze(args.data_dir, max(1, args.workers))
    events.to_csv(args.output_dir / "causal_rerank_events.csv", index=False)
    (args.output_dir / "causal_rerank_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "per_env"}, indent=2), flush=True)
    print(f"Wrote {len(events)} events to {args.output_dir}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
