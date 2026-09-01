from __future__ import annotations

import unittest

from interaction_frontier import (
    enumerate_interventions,
    observe_intervention,
    rank_interventions,
    rank_interventions_outcome_only,
    rank_interventions_regime_gated,
    mouse_effect_rate,
    summarize_targets,
)


def blank(rows: int, cols: int, value: int = 0) -> list[list[int]]:
    return [[value for _ in range(cols)] for _ in range(rows)]


def paint(grid: list[list[int]], row: int, col: int, shape: list[list[int]]) -> None:
    for drow, values in enumerate(shape):
        for dcol, value in enumerate(values):
            if value >= 0:
                grid[row + drow][col + dcol] = value


class InteractionFrontierTests(unittest.TestCase):
    def test_repeated_objects_are_compact_high_rank_targets(self) -> None:
        grid = blank(16, 16)
        square = [[2, 2], [2, 2]]
        for row, col in ((3, 3), (3, 10), (10, 3)):
            paint(grid, row, col, square)

        summary = summarize_targets(grid)

        self.assertEqual(summary["background_colors"], [0])
        repeated = [
            candidate
            for candidate in summary["candidates"]
            if candidate["copies"] == 3 and "component" in candidate["roles"]
        ]
        self.assertEqual(len(repeated), 3)
        self.assertTrue(all("component" in candidate["roles"] for candidate in repeated))
        self.assertTrue(all(candidate["score"] > 0 for candidate in repeated))
        self.assertTrue(any(lattice["instances"] == 3 for lattice in summary["lattices"]))

    def test_enclosure_proposes_negative_space_not_only_colored_pixels(self) -> None:
        grid = blank(12, 12)
        for index in range(3, 9):
            grid[3][index] = 4
            grid[8][index] = 4
            grid[index][3] = 4
            grid[index][8] = 4

        summary = summarize_targets(grid)
        center = next(
            candidate
            for candidate in summary["candidates"]
            if candidate["row"] == 5 and candidate["col"] == 5
        )

        self.assertIn("enclosed_region", center["roles"])
        self.assertEqual(grid[center["row"]][center["col"]], 0)

    def test_long_edge_hud_band_is_not_a_target(self) -> None:
        grid = blank(16, 16)
        for col in range(12):
            grid[0][col] = 7

        summary = summarize_targets(grid)

        self.assertFalse(
            any(candidate["bbox"] == [0, 0, 0, 11] for candidate in summary["candidates"])
        )

    def test_sparse_fragments_form_one_bounded_motif_target(self) -> None:
        grid = blank(20, 20)
        for row, col in ((6, 8), (6, 10), (8, 6), (8, 12), (10, 8), (10, 10)):
            grid[row][col] = 5

        summary = summarize_targets(grid)
        motif = next(
            candidate
            for candidate in summary["candidates"]
            if "motif_cluster" in candidate["roles"]
        )

        self.assertEqual((motif["row"], motif["col"]), (8, 9))
        self.assertEqual(motif["size"], 6)

    def test_mixed_controls_share_one_frontier(self) -> None:
        grid = blank(12, 12)
        paint(grid, 4, 4, [[3, 3], [3, 3]])

        frontier = enumerate_interventions(grid, ["UP", "MOUSE", "ACTION5"])
        actions = {intervention["action"] for intervention in frontier["interventions"]}

        self.assertEqual(actions, {"UP", "MOUSE", "ACTION5"})
        self.assertTrue(any(item["intervention_class"] == "control:UP" for item in frontier["interventions"]))
        self.assertTrue(
            all(
                "row" in item and "col" in item
                for item in frontier["interventions"]
                if item["action"] == "MOUSE"
            )
        )

    def test_transition_scope_is_relative_to_the_intervention(self) -> None:
        before = blank(10, 10)
        after = blank(10, 10)
        after[5][5] = 2
        local = observe_intervention(
            before,
            after,
            {"id": "I0", "action": "MOUSE", "row": 5, "col": 5, "intervention_class": "x"},
        )
        remote = observe_intervention(
            before,
            after,
            {"id": "I1", "action": "MOUSE", "row": 0, "col": 0, "intervention_class": "y"},
        )

        self.assertEqual(local["scope"], "target_local")
        self.assertEqual(remote["scope"], "remote")
        self.assertTrue(local["followup_warranted"])

    def test_exact_no_change_intervention_is_demoted_without_banning_its_class(self) -> None:
        grid = blank(16, 16)
        square = [[2, 2], [2, 2]]
        for row, col in ((3, 3), (3, 10), (10, 3)):
            paint(grid, row, col, square)
        frontier = enumerate_interventions(grid, ["MOUSE"])
        repeated = [
            intervention
            for intervention in frontier["interventions"]
            if intervention.get("target_class", "").startswith("component:")
            and "component" in intervention.get("target_roles", [])
        ]
        self.assertGreaterEqual(len(repeated), 3)
        attempted = repeated[0]
        observation = {
            "before_hash": frontier["state_hash"],
            "intervention_class": attempted["intervention_class"],
            "target_id": attempted["target_id"],
            "changed_cells": 0,
        }

        ranked = rank_interventions(frontier, [observation])

        self.assertNotEqual(ranked[0]["target_id"], attempted["target_id"])
        self.assertTrue(
            any(
                item["target_class"] == attempted["target_class"]
                and item["target_id"] != attempted["target_id"]
                for item in ranked[:5]
            )
        )

    def test_tiny_remote_object_click_can_arm_latent_followup(self) -> None:
        before = blank(12, 12)
        paint(before, 5, 5, [[3, 3], [3, 3]])
        after = [row[:] for row in before]
        after[0][0] = 7
        observation = observe_intervention(
            before, after,
            {"action": "MOUSE", "row": 5, "col": 5, "target_roles": ["component"], "intervention_class": "mouse:test"},
        )
        self.assertEqual(observation["causality"], "uncertain_ambient")
        self.assertTrue(observation["latent_selection"])
        self.assertTrue(observation["followup_warranted"])
        self.assertEqual(observation["followup_reason"], "latent_selection")

    def test_interior_revisit_demotes_same_coordinate_despite_hud_change(self) -> None:
        base = blank(16, 16)
        for row, col in ((4, 4), (4, 10), (10, 4)):
            paint(base, row, col, [[2, 2], [2, 2]])
        first = enumerate_interventions(base, ["MOUSE"])
        attempted = rank_interventions(first, [])[0]
        changed = [row[:] for row in base]
        changed[0][0] = 7
        observation = observe_intervention(base, changed, attempted)
        current = enumerate_interventions(changed, ["MOUSE"])
        ranked = rank_interventions(current, [observation])
        self.assertNotEqual(
            (ranked[0]["row"], ranked[0]["col"]),
            (attempted["row"], attempted["col"]),
        )

    def test_mouse_effect_rate_uses_only_mouse_observations(self) -> None:
        observations = [
            {"intervention_class": "control:UP", "changed_cells": 0},
            {"intervention_class": "control:MOUSE->x", "changed_cells": 3},
            {"intervention_class": "control:MOUSE->y", "changed_cells": 0},
        ]
        self.assertEqual(mouse_effect_rate(observations), 0.5)

    def test_regime_gate_preserves_static_order_below_threshold(self) -> None:
        grid = blank(16, 16)
        for row, col in ((3, 3), (3, 10), (10, 3)):
            paint(grid, row, col, [[2, 2], [2, 2]])
        frontier = enumerate_interventions(grid, ["MOUSE"])
        observations = [{"intervention_class": "control:MOUSE->x", "changed_cells": 0}]
        ranked = rank_interventions_regime_gated(frontier, observations)
        self.assertEqual(
            [item["target_id"] for item in ranked],
            [item["target_id"] for item in frontier["interventions"]],
        )
        self.assertTrue(all(item["regime_policy"] == "static" for item in ranked))

    def test_regime_gate_matches_outcome_only_above_threshold(self) -> None:
        grid = blank(16, 16)
        for row, col in ((3, 3), (3, 10), (10, 3)):
            paint(grid, row, col, [[2, 2], [2, 2]])
        frontier = enumerate_interventions(grid, ["MOUSE"])
        productive_class = frontier["interventions"][1]["intervention_class"]
        observations = [
            {"intervention_class": productive_class, "changed_cells": 2}
            for _ in range(64)
        ]
        expected = rank_interventions_outcome_only(frontier, observations)
        ranked = rank_interventions_regime_gated(frontier, observations)
        self.assertEqual(
            [item["intervention_key"] for item in ranked],
            [item["intervention_key"] for item in expected],
        )
        self.assertTrue(all(item["regime_policy"] == "outcome_only" for item in ranked))

    def test_output_is_deterministic(self) -> None:
        grid = blank(14, 14)
        paint(grid, 2, 2, [[1, 2], [2, 1]])
        paint(grid, 8, 8, [[1, 2], [2, 1]])
        self.assertEqual(summarize_targets(grid), summarize_targets(grid))


if __name__ == "__main__":
    unittest.main()
