from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from regime_gate_loeo import KEYS, add_history_features

THRESHOLD = 61 / 62


def load_events(causal_path: Path, human_path: Path) -> pd.DataFrame:
    causal = pd.read_csv(causal_path)
    human = pd.read_csv(human_path)
    cols = KEYS + ["changed_cells", "effectful", "objective_progress", "candidate_count"]
    merged = causal.merge(human[cols], on=KEYS, how="left", validate="one_to_one")
    if merged["effectful"].isna().any():
        raise AssertionError("human replay merge left missing effect labels")
    return add_history_features(merged)


def summarize(events: pd.DataFrame) -> dict:
    covered = events.loc[events["covered"]].copy()
    covered["gate"] = covered["effect_rate"].ge(THRESHOLD)
    covered["fixed_hit"] = np.where(covered["gate"], covered["outcome_hit"], covered["static_hit"])
    rows = []
    for env, group in covered.groupby("env", sort=True):
        static = float(group["static_hit"].mean())
        outcome = float(group["outcome_hit"].mean())
        fixed = float(group["fixed_hit"].mean())
        rows.append({
            "env": env,
            "covered_events": int(len(group)),
            "static_top8": static,
            "outcome_top8": outcome,
            "fixed_top8": fixed,
            "delta_vs_static": fixed - static,
            "gate_activation_pct": float(group["gate"].mean()),
        })
    frame = pd.DataFrame(rows)
    return {
        "threshold": THRESHOLD,
        "threshold_fraction": "61/62",
        "environments": int(len(frame)),
        "covered_events": int(len(covered)),
        "macro_static": float(frame["static_top8"].mean()),
        "macro_outcome": float(frame["outcome_top8"].mean()),
        "macro_fixed": float(frame["fixed_top8"].mean()),
        "macro_delta_vs_static": float(frame["delta_vs_static"].mean()),
        "event_weighted_static": float(covered["static_hit"].mean()),
        "event_weighted_outcome": float(covered["outcome_hit"].mean()),
        "event_weighted_fixed": float(covered["fixed_hit"].mean()),
        "macro_gate_activation_pct": float(frame["gate_activation_pct"].mean()),
        "improved_envs": int((frame["delta_vs_static"] > 0).sum()),
        "hurt_envs": int((frame["delta_vs_static"] < 0).sum()),
        "unchanged_envs": int((frame["delta_vs_static"] == 0).sum()),
        "per_env": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal-events", type=Path, required=True)
    parser.add_argument("--human-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(load_events(args.causal_events, args.human_events))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_env"}, indent=2))
    print("WROTE", args.output)


if __name__ == "__main__":
    main()
