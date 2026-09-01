"""Evaluate visual interaction proposals against local public ARC environments.

The validator is intentionally outside the production solver.  It may inspect
engine metadata to measure proposal coverage, but the proposal generator itself
receives only the rendered grid and legal action names.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable

from arcengine import ARCBaseGame, ActionInput, GameAction

from interaction_frontier import enumerate_interventions, observe_intervention
from mouse_hint import summarize_mouse


def _discover(environment_root: Path, requested: set[str]) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for game_dir in sorted(path for path in environment_root.iterdir() if path.is_dir()):
        game_id = game_dir.name
        if requested and game_id not in requested:
            continue
        sources = sorted(game_dir.glob("*/*.py"))
        if sources:
            found.append((game_id, sources[0]))
    return found


def _load_game(game_id: str, source: Path) -> type[ARCBaseGame]:
    module_name = f"interaction_frontier_env_{game_id}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    classes = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and value.__module__ == module.__name__
        and issubclass(value, ARCBaseGame)
        and value is not ARCBaseGame
    ]
    if len(classes) != 1:
        raise RuntimeError(f"Expected one game class in {source}, found {len(classes)}")
    return classes[0]


def _render(game: ARCBaseGame) -> list[list[int]]:
    return game.camera.render(game.current_level.get_sprites()).tolist()


def _mouse_available(game: ARCBaseGame) -> bool:
    return 6 in {int(value) for value in game._available_actions}  # noqa: SLF001 - offline oracle


def _clickable_regions(game: ARCBaseGame) -> list[tuple[int, int, int, int]]:
    scale, x_offset, y_offset = game.camera._calculate_scale_and_offset()  # noqa: SLF001
    sprites = list(game.current_level.get_sprites_by_tag("sys_click"))
    sprites.extend(game.current_level.get_sprites_by_tag("sys_place"))
    regions: list[tuple[int, int, int, int]] = []
    for sprite in sprites:
        if not game._is_sprite_clickable_now(sprite):  # noqa: SLF001
            continue
        rendered = sprite.render()
        ys, xs = (rendered >= 0).nonzero()
        if not len(xs):
            continue
        regions.append(
            (
                int((sprite.x + min(xs)) * scale + x_offset),
                int((sprite.y + min(ys)) * scale + y_offset),
                int((sprite.x + max(xs) + 1) * scale + x_offset - 1),
                int((sprite.y + max(ys) + 1) * scale + y_offset - 1),
            )
        )
    return regions


def _covered(point: tuple[int, int], regions: Iterable[tuple[int, int, int, int]]) -> bool:
    row, col = point
    return any(left <= col <= right and top <= row <= bottom for left, top, right, bottom in regions)


def _coverage(
    points: list[tuple[int, int]], regions: list[tuple[int, int, int, int]]
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for limit in (1, 4, 8, len(points)):
        prefix = points[:limit]
        result[f"top{limit}"] = {
            "points_in_regions": sum(_covered(point, regions) for point in prefix),
            "unique_regions": sum(
                any(left <= col <= right and top <= row <= bottom for row, col in prefix)
                for left, top, right, bottom in regions
            ),
        }
    result["all"] = {
        "points_in_regions": sum(_covered(point, regions) for point in points),
        "unique_regions": sum(
            any(left <= col <= right and top <= row <= bottom for row, col in points)
            for left, top, right, bottom in regions
        ),
    }
    return result


def _counterfactual(
    game_class: type[ARCBaseGame], intervention: dict[str, Any]
) -> dict[str, Any]:
    game = game_class()
    game.full_reset()
    before = _render(game)
    result = game.perform_action(
        ActionInput(
            id=GameAction.ACTION6,
            data={"x": int(intervention["col"]), "y": int(intervention["row"])},
        )
    )
    after = result.frame[-1] if result.frame else before
    observation = observe_intervention(
        before,
        after,
        intervention,
        level_completed=bool(getattr(result, "levels_completed", 0)),
        terminal=str(getattr(result, "state", "")).upper().endswith(("WIN", "GAME_OVER")),
    )
    return {
        "point": [intervention["row"], intervention["col"]],
        "target_id": intervention.get("target_id"),
        "roles": intervention.get("target_roles"),
        "changed_cells": observation.get("changed_cells"),
        "scope": observation.get("scope"),
        "motions": observation.get("motions"),
        "transition_class": observation.get("transition_class"),
        "levels_completed": int(getattr(result, "levels_completed", 0) or 0),
        "state": str(getattr(result, "state", "")),
    }


def evaluate(game_id: str, source: Path, max_candidates: int) -> dict[str, Any]:
    game_class = _load_game(game_id, source)
    game = game_class()
    game.full_reset()
    if not _mouse_available(game):
        return {"game": game_id, "mouse": False}
    grid = _render(game)
    frontier = enumerate_interventions(grid, ["MOUSE"], max_targets=max_candidates)
    mouse = [item for item in frontier["interventions"] if item["action"] == "MOUSE"]
    legacy = summarize_mouse(grid, max_candidates=max_candidates).get("candidates", [])
    regions = _clickable_regions(game)
    new_points = [(int(item["row"]), int(item["col"])) for item in mouse]
    legacy_points = [
        (int(item["center"][0]), int(item["center"][1]))
        for item in legacy
        if isinstance(item.get("center"), list) and len(item["center"]) == 2
    ]
    transitions = [_counterfactual(game_class, item) for item in mouse[:max_candidates]]
    legacy_interventions = [
        {
            "id": f"L{index}",
            "action": "MOUSE",
            "row": point[0],
            "col": point[1],
            "target_id": f"L{index}",
            "target_roles": ["legacy_component_center"],
            "intervention_class": "legacy",
        }
        for index, point in enumerate(legacy_points)
    ]
    legacy_transitions = [
        _counterfactual(game_class, item) for item in legacy_interventions[:max_candidates]
    ]
    return {
        "game": game_id,
        "mouse": True,
        "shape": [len(grid), len(grid[0]) if grid else 0],
        "engine_clickable_regions": len(regions),
        "new_candidates": len(mouse),
        "legacy_candidates": len(legacy_points),
        "new_clickable_coverage": _coverage(new_points, regions),
        "legacy_clickable_coverage": _coverage(legacy_points, regions),
        "new_role_counts": {
            role: sum(role in item.get("target_roles", []) for item in mouse)
            for role in sorted({role for item in mouse for role in item.get("target_roles", [])})
        },
        "new_candidate_metadata": frontier["targets"]["candidates"][:max_candidates],
        "counterfactuals": transitions,
        "legacy_counterfactuals": legacy_transitions,
        "nonzero_transitions": sum(int(item.get("changed_cells") or 0) > 0 for item in transitions),
        "legacy_nonzero_transitions": sum(
            int(item.get("changed_cells") or 0) > 0 for item in legacy_transitions
        ),
        "target_local_transitions": sum(item.get("scope") == "target_local" for item in transitions),
        "level_completions": sum(int(item.get("levels_completed") or 0) > 0 for item in transitions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("environment_root", type=Path)
    parser.add_argument("--games", nargs="*", default=[])
    parser.add_argument("--max-candidates", type=int, default=16)
    args = parser.parse_args()
    requested = {str(value) for value in args.games}
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for game_id, source in _discover(args.environment_root, requested):
        try:
            result = evaluate(game_id, source, max(1, args.max_candidates))
        except Exception as exc:  # keep the broad public sweep diagnostic
            failures.append({"game": game_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if result.get("mouse"):
            results.append(result)
    print(json.dumps({"results": results, "failures": failures}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
