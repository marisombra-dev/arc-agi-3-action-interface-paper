from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KEYS = ["env", "guid", "event_index"]
FEATURES = [
    "candidate_count16", "history_count", "mouse_history_count", "mouse_density",
    "effect_rate", "objective_rate", "mean_log_changed", "recent4_effect_rate",
    "recent8_effect_rate", "last_effect", "nochange_streak", "effect_streak",
    "same_state_attempts", "state_diversity", "action_gap", "level_before",
]
def _safe_rate(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _streak(values: list[bool], wanted: bool) -> int:
    count = 0
    for value in reversed(values):
        if bool(value) != wanted:
            break
        count += 1
    return count


def load_events(dev_path: Path, held_path: Path, human_path: Path) -> pd.DataFrame:
    dev = pd.read_csv(dev_path)
    held = pd.read_csv(held_path)
    causal = pd.concat([dev, held], ignore_index=True)
    human = pd.read_csv(human_path)
    cols = KEYS + ["changed_cells", "effectful", "objective_progress", "candidate_count"]
    merged = causal.merge(human[cols], on=KEYS, how="left", validate="one_to_one")
    if merged["effectful"].isna().any():
        raise AssertionError("human replay merge left missing effect labels")
    return merged
def add_history_features(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (_, _), group in events.groupby(["env", "guid"], sort=False):
        group = group.sort_values("event_index")
        history: deque[dict[str, Any]] = deque(maxlen=64)
        prior_event = 0
        for _, source in group.iterrows():
            event_index = int(source["event_index"])
            gap = max(0, event_index - prior_event - 1)
            for _ in range(gap):
                history.append({"mouse": False})
            mouse_hist = [item for item in history if item.get("mouse")]
            effects = [bool(item["effectful"]) for item in mouse_hist]
            objectives = [bool(item["objective"]) for item in mouse_hist]
            changes = [np.log1p(float(item["changed"])) for item in mouse_hist]
            states = [str(item["state"]) for item in mouse_hist]
            current_state = str(source["before_key"])
            row = source.to_dict()
            row.update({
                "candidate_count16": min(16, int(source["candidate_count"])),
                "history_count_rebuilt": len(history),
                "mouse_history_count_rebuilt": len(mouse_hist),
                "mouse_density": len(mouse_hist) / len(history) if history else 0.0,
                "effect_rate": _safe_rate([float(v) for v in effects]),
                "objective_rate": _safe_rate([float(v) for v in objectives]),
                "mean_log_changed": _safe_rate(changes),
                "recent4_effect_rate": _safe_rate([float(v) for v in effects[-4:]]),
                "recent8_effect_rate": _safe_rate([float(v) for v in effects[-8:]]),
                "last_effect": float(effects[-1]) if effects else 0.0,
                "nochange_streak": _streak(effects, False),
                "effect_streak": _streak(effects, True),
                "same_state_attempts": sum(state == current_state for state in states),
                "state_diversity": len(set(states)) / len(states) if states else 0.0,
                "action_gap": gap,
            })
            rows.append(row)
            history.append({"mouse": True, "effectful": bool(source["effectful"]),
                            "objective": bool(source["objective_progress"]),
                            "changed": float(source["changed_cells"]), "state": current_state})
            prior_event = event_index
    out = pd.DataFrame(rows)
    if not np.array_equal(out["history_count_rebuilt"].to_numpy(), out["history_count"].to_numpy()):
        raise AssertionError("rebuilt 64-action history length diverges from v47 evaluator")
    if not np.array_equal(out["mouse_history_count_rebuilt"].to_numpy(), out["mouse_history_count"].to_numpy()):
        raise AssertionError("rebuilt mouse-history count diverges from v47 evaluator")
    out["covered"] = out["static_rank"].notna()
    out["static_hit"] = out["static_rank"].le(8)
    out["outcome_hit"] = out["outcome_only_rank"].le(8)
    out["utility"] = out["outcome_hit"].astype(int) - out["static_hit"].astype(int)
    return out


def _macro_score(frame: pd.DataFrame, use_outcome: np.ndarray) -> float:
    scored = frame.loc[frame["covered"]].copy()
    mask = np.asarray(use_outcome)[frame["covered"].to_numpy()]
    hit = np.where(mask, scored["outcome_hit"].to_numpy(), scored["static_hit"].to_numpy())
    scored["policy_hit"] = hit
    return float(scored.groupby("env")["policy_hit"].mean().mean())
def _fit_stump(train: pd.DataFrame) -> dict[str, Any]:
    best = {"feature": "STATIC", "direction": "never", "threshold": 0.0,
            "train_macro": _macro_score(train, np.zeros(len(train), dtype=bool))}
    outcome_macro = _macro_score(train, np.ones(len(train), dtype=bool))
    if outcome_macro > best["train_macro"]:
        best = {"feature": "OUTCOME", "direction": "always", "threshold": 0.0,
                "train_macro": outcome_macro}
    covered = train.loc[train["covered"]]
    for feature in FEATURES:
        values = covered[feature].to_numpy(dtype=float)
        if len(np.unique(values)) < 2:
            continue
        thresholds = np.unique(np.quantile(values, np.linspace(0.05, 0.95, 19)))
        for threshold in thresholds:
            for direction in ("ge", "le"):
                raw = train[feature].to_numpy(dtype=float)
                mask = raw >= threshold if direction == "ge" else raw <= threshold
                score = _macro_score(train, mask)
                if score > best["train_macro"] + 1e-12:
                    best = {"feature": feature, "direction": direction,
                            "threshold": float(threshold), "train_macro": score}
    return best
def _apply_rule(frame: pd.DataFrame, rule: dict[str, Any]) -> np.ndarray:
    if rule["direction"] == "never":
        return np.zeros(len(frame), dtype=bool)
    if rule["direction"] == "always":
        return np.ones(len(frame), dtype=bool)
    raw = frame[rule["feature"]].to_numpy(dtype=float)
    if rule["direction"] == "ge":
        return raw >= float(rule["threshold"])
    return raw <= float(rule["threshold"])


def evaluate_loeo(events: pd.DataFrame) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for env in sorted(events["env"].unique()):
        train = events.loc[events["env"] != env].reset_index(drop=True)
        test = events.loc[events["env"] == env].reset_index(drop=True)
        rule = _fit_stump(train)
        gate = _apply_rule(test, rule)
        covered = test["covered"].to_numpy()
        static = float(test.loc[test["covered"], "static_hit"].mean())
        outcome = float(test.loc[test["covered"], "outcome_hit"].mean())
        gated = float(np.where(gate[covered], test.loc[test["covered"], "outcome_hit"],
                               test.loc[test["covered"], "static_hit"]).mean())
        folds.append({"env": env, "covered_events": int(covered.sum()),
                      "static_top8": static, "outcome_top8": outcome,
                      "gated_top8": gated, "delta_vs_static": gated - static, **rule})
    frame = pd.DataFrame(folds)
    return {
        "folds": folds,
        "macro_static": float(frame["static_top8"].mean()),
        "macro_outcome": float(frame["outcome_top8"].mean()),
        "macro_gated": float(frame["gated_top8"].mean()),
        "macro_delta_vs_static": float(frame["delta_vs_static"].mean()),
        "improved_envs": int((frame["delta_vs_static"] > 1e-12).sum()),
        "hurt_envs": int((frame["delta_vs_static"] < -1e-12).sum()),
        "unchanged_envs": int((frame["delta_vs_static"].abs() <= 1e-12).sum()),
        "selected_features": frame["feature"].value_counts().to_dict(),
    }
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-events", type=Path, required=True)
    parser.add_argument("--heldout-events", type=Path, required=True)
    parser.add_argument("--human-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("paper_prize/results/regime_gate_loeo.json"))
    args = parser.parse_args()

    events = add_history_features(load_events(args.dev_events, args.heldout_events, args.human_events))
    result = evaluate_loeo(events)
    result["events"] = int(len(events))
    result["covered_events"] = int(events["covered"].sum())
    result["environments"] = int(events["env"].nunique())
    oracle = events.loc[events["covered"]].groupby("env").apply(
        lambda g: max(float(g["static_hit"].mean()), float(g["outcome_hit"].mean())),
        include_groups=False,
    )
    result["macro_environment_oracle"] = float(oracle.mean())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "folds"}, indent=2, sort_keys=True))
    for fold in result["folds"]:
        print(fold)


if __name__ == "__main__":
    main()
