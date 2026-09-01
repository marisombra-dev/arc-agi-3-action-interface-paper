from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from regime_gate_loeo import add_history_features, load_events


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dev-events", type=Path, required=True)
    p.add_argument("--heldout-events", type=Path, required=True)
    p.add_argument("--human-events", type=Path, required=True)
    p.add_argument("--loeo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    events = add_history_features(load_events(a.dev_events, a.heldout_events, a.human_events))
    result = json.loads(a.loeo.read_text(encoding="utf-8"))
    rows = []
    for fold in result["folds"]:
        env = fold["env"]
        group = events.loc[(events["env"] == env) & events["covered"]].copy()
        threshold = float(fold["threshold"])
        active = group["effect_rate"] >= threshold
        static = group["static_hit"].astype(bool)
        outcome = group["outcome_hit"].astype(bool)
        utility = outcome.astype(int) - static.astype(int)
        rows.append({
            "env": env,
            "covered_events": int(len(group)),
            "gate_activation_pct": float(active.mean()),
            "beneficial_switch_pct": float((active & (utility > 0)).mean()),
            "harmful_switch_pct": float((active & (utility < 0)).mean()),
            "neutral_switch_pct": float((active & (utility == 0)).mean()),
            "no_history_pct": float((group["mouse_history_count"] == 0).mean()),
            "median_mouse_history": float(group["mouse_history_count"].median()),
        })
    frame = pd.DataFrame(rows)
    summary = {
        "macro_gate_activation_pct": float(frame["gate_activation_pct"].mean()),
        "macro_beneficial_switch_pct": float(frame["beneficial_switch_pct"].mean()),
        "macro_harmful_switch_pct": float(frame["harmful_switch_pct"].mean()),
        "macro_neutral_switch_pct": float(frame["neutral_switch_pct"].mean()),
        "folds": rows,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k != "folds"}, indent=2))
    print(frame[["env","gate_activation_pct","beneficial_switch_pct","harmful_switch_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
