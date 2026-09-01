"""Source-free interaction proposals and transition-class memory.

This is an offline prototype.  It deliberately does not know game ids, rules,
solutions, or engine internals.  Its job is to turn a visible grid and the
currently legal controls into a compact intervention frontier that works for
both ordinary actions and coordinate actions.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict, deque
from typing import Any, Iterable, Mapping, Sequence

Grid = Sequence[Sequence[int]]
Point = tuple[int, int]


def grid_fingerprint(grid: Grid) -> str:
    digest = hashlib.blake2s(digest_size=8)
    for row in grid:
        digest.update(bytes(int(value) & 0xFF for value in row))
        digest.update(b"\xff")
    return digest.hexdigest()


def _rectangular(grid: Grid) -> tuple[tuple[int, ...], ...]:
    rows = tuple(tuple(int(value) for value in row) for row in grid)
    if not rows:
        return ()
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        return ()
    return rows


def interior_fingerprint(grid: Grid, *, margin: int = 3) -> str:
    """HUD-tolerant fingerprint used only as a soft revisit signal."""
    normalized = _rectangular(grid)
    if not normalized:
        return grid_fingerprint(grid)
    rows, cols = len(normalized), len(normalized[0])
    margin = max(0, int(margin))
    if margin == 0 or rows <= 2 * margin or cols <= 2 * margin:
        return grid_fingerprint(normalized)
    core = tuple(tuple(row[margin:cols-margin]) for row in normalized[margin:rows-margin])
    return grid_fingerprint(core)


def intervention_key(intervention: Mapping[str, Any]) -> str:
    action = str(intervention.get("action") or "").upper()
    if action == "MOUSE":
        return f"MOUSE:{int(intervention.get('row') or 0)}:{int(intervention.get('col') or 0)}"
    return action


def _component_records(grid: tuple[tuple[int, ...], ...]) -> list[dict[str, Any]]:
    rows, cols = len(grid), len(grid[0])
    seen: set[Point] = set()
    records: list[dict[str, Any]] = []
    for row in range(rows):
        for col in range(cols):
            if (row, col) in seen:
                continue
            color = grid[row][col]
            stack = [(row, col)]
            seen.add((row, col))
            points: list[Point] = []
            while stack:
                current = stack.pop()
                points.append(current)
                for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nxt = current[0] + drow, current[1] + dcol
                    if (
                        0 <= nxt[0] < rows
                        and 0 <= nxt[1] < cols
                        and nxt not in seen
                        and grid[nxt[0]][nxt[1]] == color
                    ):
                        seen.add(nxt)
                        stack.append(nxt)
            points.sort()
            min_row = min(point[0] for point in points)
            min_col = min(point[1] for point in points)
            max_row = max(point[0] for point in points)
            max_col = max(point[1] for point in points)
            height, width = max_row - min_row + 1, max_col - min_col + 1
            normalized = tuple((r - min_row, c - min_col) for r, c in points)
            edge_sides = (
                int(min_row == 0)
                + int(min_col == 0)
                + int(max_row == rows - 1)
                + int(max_col == cols - 1)
            )
            mean_row = sum(point[0] for point in points) / len(points)
            mean_col = sum(point[1] for point in points) / len(points)
            medoid = min(
                points,
                key=lambda point: (
                    (point[0] - mean_row) ** 2 + (point[1] - mean_col) ** 2,
                    point,
                ),
            )
            records.append(
                {
                    "color": color,
                    "points": tuple(points),
                    "point_set": frozenset(points),
                    "size": len(points),
                    "bbox": (min_row, min_col, max_row, max_col),
                    "height": height,
                    "width": width,
                    "fill": len(points) / (height * width),
                    "shape": normalized,
                    "medoid": medoid,
                    "edge_sides": edge_sides,
                }
            )
    return records


def _stable_signature(value: object) -> str:
    return hashlib.blake2s(repr(value).encode("utf-8"), digest_size=6).hexdigest()


def _canonical_shape(shape: tuple[Point, ...], bbox: tuple[int, int]) -> tuple[Point, ...]:
    """Collapse uniformly pixel-scaled shapes for legacy-safe repetition ranking."""
    points = set(shape)
    height, width = bbox
    for scale in range(min(4, height, width), 1, -1):
        if height % scale or width % scale or len(points) % (scale * scale):
            continue
        coarse: set[Point] = set()
        valid = True
        for row in range(0, height, scale):
            for col in range(0, width, scale):
                block = {(row + dr, col + dc) for dr in range(scale) for dc in range(scale)}
                hit = len(block & points)
                if hit not in (0, scale * scale):
                    valid = False
                    break
                if hit:
                    coarse.add((row // scale, col // scale))
            if not valid:
                break
        if valid and coarse:
            return tuple(sorted(coarse))
    return tuple(shape)


def _large_backgrounds(
    components: Sequence[Mapping[str, Any]], rows: int, cols: int
) -> tuple[set[int], set[int]]:
    area = rows * cols
    background_indices: set[int] = set()
    colors: set[int] = set()
    for index, component in enumerate(components):
        size = int(component["size"])
        height = int(component["height"])
        width = int(component["width"])
        edge = int(component["edge_sides"])
        large_connected_field = size >= max(16, math.ceil(area * 0.12)) and edge
        scene_spanning = size >= max(16, math.ceil(area * 0.06)) and (
            height >= math.ceil(rows * 0.75) or width >= math.ceil(cols * 0.75)
        )
        dominant = size >= math.ceil(area * 0.45)
        if large_connected_field or scene_spanning or dominant:
            background_indices.add(index)
            colors.add(int(component["color"]))
    return background_indices, colors


def _edge_band(component: Mapping[str, Any], rows: int, cols: int) -> bool:
    if not component["edge_sides"]:
        return False
    height, width = int(component["height"]), int(component["width"])
    return bool(
        (height <= 2 and width >= max(8, math.ceil(cols * 0.25)))
        or (width <= 2 and height >= max(8, math.ceil(rows * 0.25)))
    )


def _foreground_islands(
    grid: tuple[tuple[int, ...], ...], background_points: set[Point]
) -> list[dict[str, Any]]:
    """Return 8-connected, possibly multicolor foreground islands."""
    rows, cols = len(grid), len(grid[0])
    foreground = {
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if (row, col) not in background_points
    }
    seen: set[Point] = set()
    islands: list[dict[str, Any]] = []
    for seed in sorted(foreground):
        if seed in seen:
            continue
        stack = [seed]
        seen.add(seed)
        points: list[Point] = []
        while stack:
            point = stack.pop()
            points.append(point)
            for drow in (-1, 0, 1):
                for dcol in (-1, 0, 1):
                    if not (drow or dcol):
                        continue
                    nxt = point[0] + drow, point[1] + dcol
                    if nxt in foreground and nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
        colors = {grid[row][col] for row, col in points}
        if len(colors) < 2:
            continue
        min_row = min(point[0] for point in points)
        min_col = min(point[1] for point in points)
        max_row = max(point[0] for point in points)
        max_col = max(point[1] for point in points)
        height, width = max_row - min_row + 1, max_col - min_col + 1
        if len(points) > 512 or height > 32 or width > 32:
            continue
        mean_row = sum(point[0] for point in points) / len(points)
        mean_col = sum(point[1] for point in points) / len(points)
        medoid = min(
            points,
            key=lambda point: (
                (point[0] - mean_row) ** 2 + (point[1] - mean_col) ** 2,
                point,
            ),
        )
        pattern = tuple(
            sorted((row - min_row, col - min_col, grid[row][col]) for row, col in points)
        )
        islands.append(
            {
                "points": tuple(sorted(points)),
                "medoid": medoid,
                "bbox": (min_row, min_col, max_row, max_col),
                "height": height,
                "width": width,
                "size": len(points),
                "pattern": pattern,
                "edge_sides": int(min_row == 0)
                + int(min_col == 0)
                + int(max_row == rows - 1)
                + int(max_col == cols - 1),
            }
        )
    return islands


def _gcd_step(values: Iterable[int]) -> int | None:
    ordered = sorted(set(int(value) for value in values))
    step = 0
    for left, right in zip(ordered, ordered[1:]):
        step = math.gcd(step, right - left)
    return step or None


def _motif_clusters(
    components: Sequence[Mapping[str, Any]],
    background_indices: set[int],
    rows: int,
    cols: int,
) -> list[dict[str, Any]]:
    """Group small nearby color fragments into bounded visual motifs.

    A sprite can be a sparse outline or use several colors, so same-color
    connected components alone are not an adequate object vocabulary.
    """
    eligible = [
        index
        for index, component in enumerate(components)
        if index not in background_indices
        and int(component["size"]) <= 64
        and int(component["height"]) <= 12
        and int(component["width"]) <= 12
        and not _edge_band(component, rows, cols)
    ]
    parent = {index: index for index in eligible}
    cluster_bbox = {index: tuple(components[index]["bbox"]) for index in eligible}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        left_box, right_box = cluster_bbox[left_root], cluster_bbox[right_root]
        merged = (
            min(left_box[0], right_box[0]),
            min(left_box[1], right_box[1]),
            max(left_box[2], right_box[2]),
            max(left_box[3], right_box[3]),
        )
        if merged[2] - merged[0] + 1 > 12 or merged[3] - merged[1] + 1 > 12:
            return
        parent[right_root] = left_root
        cluster_bbox[left_root] = merged

    for position, left_index in enumerate(eligible):
        left = components[left_index]
        left_top, left_col, left_bottom, left_right = left["bbox"]
        for right_index in eligible[position + 1 :]:
            right = components[right_index]
            right_top, right_col, right_bottom, right_right = right["bbox"]
            row_gap = max(0, left_top - right_bottom - 1, right_top - left_bottom - 1)
            col_gap = max(0, left_col - right_right - 1, right_col - left_right - 1)
            if max(row_gap, col_gap) <= 1:
                union(left_index, right_index)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in eligible:
        groups[find(index)].append(index)
    motifs: list[dict[str, Any]] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        points = [
            (row, col, int(components[index]["color"]))
            for index in indices
            for row, col in components[index]["points"]
        ]
        min_row = min(point[0] for point in points)
        min_col = min(point[1] for point in points)
        max_row = max(point[0] for point in points)
        max_col = max(point[1] for point in points)
        height, width = max_row - min_row + 1, max_col - min_col + 1
        if height > 12 or width > 12 or len(points) > 256:
            continue
        pattern = tuple(
            sorted((row - min_row, col - min_col, color) for row, col, color in points)
        )
        mean_row = sum(point[0] for point in points) / len(points)
        mean_col = sum(point[1] for point in points) / len(points)
        medoid = min(
            ((point[0], point[1]) for point in points),
            key=lambda point: (
                (point[0] - mean_row) ** 2 + (point[1] - mean_col) ** 2,
                point,
            ),
        )
        motifs.append(
            {
                "center": ((min_row + max_row) // 2, (min_col + max_col) // 2),
                "medoid": medoid,
                "bbox": (min_row, min_col, max_row, max_col),
                "size": len(points),
                "colors": sorted({point[2] for point in points}),
                "pattern": pattern,
                "fragments": len(indices),
                "edge_sides": int(min_row == 0)
                + int(min_col == 0)
                + int(max_row == rows - 1)
                + int(max_col == cols - 1),
            }
        )
    return motifs


def summarize_targets(grid: Grid, *, max_candidates: int = 24) -> dict[str, Any]:
    """Extract compact visual targets without deciding which one to click."""
    normalized = _rectangular(grid)
    if not normalized:
        return {"shape": [0, 0], "background_colors": [], "lattices": [], "candidates": []}
    rows, cols = len(normalized), len(normalized[0])
    area = rows * cols
    dominant_colors = {
        color
        for color, _ in Counter(value for row in normalized for value in row).most_common(2)
    }
    components = _component_records(normalized)
    background_indices, background_colors = _large_backgrounds(components, rows, cols)
    max_component = min(512, max(16, math.ceil(area * 0.20)))

    shape_counts = Counter(
        (int(component["color"]), tuple(component["shape"]))
        for index, component in enumerate(components)
        if index not in background_indices
        and int(component["size"]) <= max_component
        and int(component["height"]) <= 32
        and int(component["width"]) <= 32
        and not _edge_band(component, rows, cols)
    )
    legacy_counts = Counter(
        (int(component["color"]), _canonical_shape(tuple(component["shape"]), (int(component["height"]), int(component["width"]))))
        for index, component in enumerate(components)
        if index not in background_indices
        and 2 <= int(component["size"]) <= 256
        and int(component["height"]) <= 20
        and int(component["width"]) <= 20
        and int(component["color"]) not in dominant_colors
    )
    raw: list[dict[str, Any]] = []
    for index, component in enumerate(components):
        if index in background_indices or _edge_band(component, rows, cols):
            continue
        size = int(component["size"])
        height, width = int(component["height"]), int(component["width"])
        if size > max_component or height > 32 or width > 32:
            continue
        color = int(component["color"])
        key = (color, tuple(component["shape"]))
        copies = int(shape_counts[key])
        edge = bool(component["edge_sides"])
        score = 2 + min(6, max(0, copies - 1) * 2)
        score += 3 if not edge else -2
        score += 2 if 2 <= size <= 128 else -1
        score -= 2 if color in background_colors else 0
        signature = "component:" + _stable_signature(key)
        min_row, min_col, max_row, max_col = component["bbox"]
        medoid_row, medoid_col = component["medoid"]
        center = ((min_row + max_row) // 2, (min_col + max_col) // 2)
        legacy_eligible = bool(
            color not in dominant_colors
            and 2 <= size <= 256
            and height <= 20
            and width <= 20
        )
        legacy_key = (color, _canonical_shape(tuple(component["shape"]), (height, width)))
        legacy_copies = int(legacy_counts[legacy_key]) if legacy_eligible else 0
        legacy_rank = ([int(edge), -legacy_copies, size, min_row, min_col, center[0], center[1]] if legacy_eligible else None)
        raw.append(
            {
                "row": medoid_row,
                "col": medoid_col,
                "roles": ["component"],
                "target_class": signature,
                "bbox": [min_row, min_col, max_row, max_col],
                "size": size,
                "colors": [color],
                "copies": copies,
                "edge": edge,
                "score": score,
                "legacy_eligible": legacy_eligible and (medoid_row, medoid_col) == center,
                "legacy_copies": legacy_copies,
                "legacy_rank": legacy_rank if legacy_eligible and (medoid_row, medoid_col) == center else None,
            }
        )
        if center != (medoid_row, medoid_col) and height <= 20 and width <= 20:
            raw.append(
                {
                    "row": center[0],
                    "col": center[1],
                    "roles": ["component_center"],
                    "target_class": signature,
                    "bbox": [min_row, min_col, max_row, max_col],
                    "size": size,
                    "colors": [color],
                    "copies": copies,
                    "edge": edge,
                    "score": score + 1,
                    "legacy_eligible": legacy_eligible,
                    "legacy_copies": legacy_copies,
                    "legacy_rank": legacy_rank,
                }
            )
        if (
            height >= 3
            and width >= 3
            and float(component["fill"]) < 0.75
            and center not in component["point_set"]
        ):
            raw.append(
                {
                    "row": center[0],
                    "col": center[1],
                    "roles": ["enclosed_region"],
                    "target_class": "region:" + _stable_signature(key),
                    "bbox": [min_row, min_col, max_row, max_col],
                    "size": size,
                    "colors": [color],
                    "copies": copies,
                    "edge": edge,
                    "score": score + 1,
                    "legacy_eligible": False,
                }
            )

    # Remove only pixels belonging to large background components.  Removing an
    # entire color would erase small foreground objects that reuse that color.
    background_points = {
        point
        for index in background_indices
        for point in components[index]["points"]
    }
    islands = _foreground_islands(normalized, background_points)
    island_counts = Counter(tuple(island["pattern"]) for island in islands)
    for island in islands:
        copies = island_counts[tuple(island["pattern"])]
        min_row, min_col, max_row, max_col = island["bbox"]
        colors = sorted({value for _, _, value in island["pattern"]})
        score = 4 + min(6, max(0, copies - 1) * 2) - (2 if island["edge_sides"] else 0)
        raw.append(
            {
                "row": island["medoid"][0],
                "col": island["medoid"][1],
                "roles": ["multicolor_island"],
                "target_class": "island:" + _stable_signature(tuple(island["pattern"])),
                "bbox": [min_row, min_col, max_row, max_col],
                "size": island["size"],
                "colors": colors,
                "copies": copies,
                "edge": bool(island["edge_sides"]),
                "score": score,
            }
        )

    motifs = _motif_clusters(components, background_indices, rows, cols)
    motif_counts = Counter(tuple(motif["pattern"]) for motif in motifs)
    for motif in motifs:
        copies = motif_counts[tuple(motif["pattern"])]
        min_row, min_col, max_row, max_col = motif["bbox"]
        # A bounded motif is a stronger click unit than any one of its singleton
        # fragments, even when that motif occurs only once.
        score = 12 + min(6, max(0, copies - 1) * 2) - (2 if motif["edge_sides"] else 0)
        raw.append(
            {
                "row": motif["center"][0],
                "col": motif["center"][1],
                "roles": ["motif_cluster"],
                "target_class": "motif:" + _stable_signature(tuple(motif["pattern"])),
                "bbox": [min_row, min_col, max_row, max_col],
                "size": motif["size"],
                "colors": motif["colors"],
                "copies": copies,
                "edge": bool(motif["edge_sides"]),
                "score": score,
            }
        )
        if motif["medoid"] != motif["center"]:
            raw.append(
                {
                    "row": motif["medoid"][0],
                    "col": motif["medoid"][1],
                    "roles": ["motif_member"],
                    "target_class": "motif:" + _stable_signature(tuple(motif["pattern"])),
                    "bbox": [min_row, min_col, max_row, max_col],
                    "size": motif["size"],
                    "colors": motif["colors"],
                    "copies": copies,
                    "edge": bool(motif["edge_sides"]),
                    "score": score,
                }
            )

    by_position: dict[Point, dict[str, Any]] = {}
    for candidate in raw:
        point = int(candidate["row"]), int(candidate["col"])
        current = by_position.get(point)
        if current is None or int(candidate["score"]) > int(current["score"]):
            if current is not None:
                candidate["roles"] = sorted(set(candidate["roles"]) | set(current["roles"]))
                candidate["legacy_eligible"] = bool(
                    candidate.get("legacy_eligible") or current.get("legacy_eligible")
                )
                candidate["legacy_copies"] = max(
                    int(candidate.get("legacy_copies") or 0), int(current.get("legacy_copies") or 0)
                )
                _ranks = [rank for rank in (candidate.get("legacy_rank"), current.get("legacy_rank")) if rank is not None]
                candidate["legacy_rank"] = min(_ranks) if _ranks else None
            by_position[point] = candidate
        else:
            current["roles"] = sorted(set(current["roles"]) | set(candidate["roles"]))
            current["legacy_eligible"] = bool(
                current.get("legacy_eligible") or candidate.get("legacy_eligible")
            )
            current["legacy_copies"] = max(
                int(current.get("legacy_copies") or 0), int(candidate.get("legacy_copies") or 0)
            )
            _ranks = [rank for rank in (current.get("legacy_rank"), candidate.get("legacy_rank")) if rank is not None]
            current["legacy_rank"] = min(_ranks) if _ranks else None

    lattices: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in by_position.values():
        grouped[str(candidate["target_class"])].append(candidate)
    occupied = set(by_position)
    for target_class, group in grouped.items():
        if len(group) < 3:
            continue
        row_values = [int(item["row"]) for item in group]
        col_values = [int(item["col"]) for item in group]
        row_step, col_step = _gcd_step(row_values), _gcd_step(col_values)
        lattices.append(
            {
                "target_class": target_class,
                "instances": len(group),
                "row_step": row_step,
                "col_step": col_step,
            }
        )
        unique_rows, unique_cols = sorted(set(row_values)), sorted(set(col_values))
        lattice_slots = len(unique_rows) * len(unique_cols)
        lattice_density = len(group) / lattice_slots if lattice_slots else 0.0
        if (
            len(unique_rows) < 2
            or len(unique_cols) < 2
            or lattice_slots > 64
            or lattice_density < 0.50
            or (row_step or 0) < 2
            or (col_step or 0) < 2
        ):
            continue
        exemplar = max(group, key=lambda item: int(item["score"]))
        for point in ((row, col) for row in unique_rows for col in unique_cols):
            if point in occupied:
                continue
            occupied.add(point)
            by_position[point] = {
                "row": point[0],
                "col": point[1],
                "roles": ["lattice_gap"],
                "target_class": "lattice_gap:" + target_class,
                "bbox": [point[0], point[1], point[0], point[1]],
                "size": 0,
                "colors": list(exemplar["colors"]),
                "copies": len(group),
                "edge": point[0] in (0, rows - 1) or point[1] in (0, cols - 1),
                "score": int(exemplar["score"]) - 1,
            }

    fallback_points = [
        (rows // 2, cols // 2),
        (rows // 4, cols // 4),
        (rows // 4, (3 * cols) // 4),
        ((3 * rows) // 4, cols // 4),
        ((3 * rows) // 4, (3 * cols) // 4),
    ]
    for point in fallback_points:
        if len(by_position) >= 4 or point in by_position:
            break
        by_position[point] = {
            "row": point[0],
            "col": point[1],
            "roles": ["spatial_probe"],
            "target_class": "spatial_probe",
            "bbox": [point[0], point[1], point[0], point[1]],
            "size": 0,
            "colors": [],
            "copies": 1,
            "edge": False,
            "score": -4,
        }

    def candidate_rank(item: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            -int(item["score"]),
            bool(item["edge"]),
            -int(item["copies"]),
            abs(int(item["row"]) - (rows - 1) / 2)
            + abs(int(item["col"]) - (cols - 1) / 2),
            int(item["row"]),
            int(item["col"]),
            str(item["target_class"]),
        )

    ranked_all = sorted(by_position.values(), key=candidate_rank)
    legacy_ranked = sorted(
        (item for item in ranked_all if item.get("legacy_rank") is not None),
        key=lambda item: tuple(item["legacy_rank"]),
    )
    novel_ranked = [item for item in ranked_all if not item.get("legacy_eligible")]
    limit = max(0, int(max_candidates))
    reserve = min(len(legacy_ranked), limit if not novel_ranked else max(0, limit - 1))
    selected = list(legacy_ranked[:reserve])
    selected_points = {(int(item["row"]), int(item["col"])) for item in selected}
    for item in ranked_all:
        point = int(item["row"]), int(item["col"])
        if point in selected_points:
            continue
        selected.append(item)
        selected_points.add(point)
        if len(selected) >= limit:
            break
    candidates = sorted(selected[:limit], key=candidate_rank)
    for index, candidate in enumerate(candidates):
        candidate["id"] = f"T{index}"
    return {
        "shape": [rows, cols],
        "state_hash": grid_fingerprint(normalized),
        "interior_hash": interior_fingerprint(normalized),
        "background_colors": sorted(background_colors),
        "lattices": sorted(lattices, key=lambda item: str(item["target_class"])),
        "candidates": candidates,
    }


def enumerate_interventions(
    grid: Grid, valid_actions: Sequence[str], *, max_targets: int = 24
) -> dict[str, Any]:
    """Put discrete controls and coordinate targets on one frontier."""
    legal: list[str] = []
    for value in valid_actions:
        action = str(value).strip().upper()
        if action and action not in legal:
            legal.append(action)
    target_summary = summarize_targets(grid, max_candidates=max_targets)
    interventions: list[dict[str, Any]] = []
    for action in legal:
        if action == "MOUSE":
            continue
        interventions.append(
            {
                "action": action,
                "intervention_class": f"control:{action}",
                "target_id": None,
                "priority": 0,
            }
        )
    if "MOUSE" in legal:
        for target in target_summary["candidates"]:
            interventions.append(
                {
                    "action": "MOUSE",
                    "row": target["row"],
                    "col": target["col"],
                    "target_id": target["id"],
                    "target_class": target["target_class"],
                    "target_roles": target["roles"],
                    "target_colors": target.get("colors", []),
                    "target_bbox": target.get("bbox"),
                    "target_size": target.get("size"),
                    "target_copies": target.get("copies"),
                    "intervention_class": "control:MOUSE->" + str(target["target_class"]),
                    "priority": int(target["score"]),
                    "target_rank": int(str(target["id"])[1:]),
                }
            )
    interventions.sort(
        key=lambda item: (
            -int(item["priority"]),
            str(item["action"]),
            int(item.get("target_rank") or 0),
        )
    )
    for index, intervention in enumerate(interventions):
        intervention["id"] = f"I{index}"
    return {
        "state_hash": target_summary.get("state_hash"),
        "interior_hash": target_summary.get("interior_hash"),
        "valid_actions": legal,
        "targets": target_summary,
        "interventions": interventions,
    }


def _motion_count(before: tuple[tuple[int, ...], ...], after: tuple[tuple[int, ...], ...]) -> int:
    def indexed(grid: tuple[tuple[int, ...], ...]) -> dict[tuple[int, tuple[Point, ...]], list[Point]]:
        result: dict[tuple[int, tuple[Point, ...]], list[Point]] = defaultdict(list)
        for component in _component_records(grid):
            result[(int(component["color"]), tuple(component["shape"]))].append(
                (int(component["bbox"][0]), int(component["bbox"][1]))
            )
        return result

    left, right = indexed(before), indexed(after)
    count = 0
    for key, old_positions in left.items():
        new_positions = right.get(key, [])
        if len(old_positions) == len(new_positions) == 1 and old_positions[0] != new_positions[0]:
            count += 1
    return count


def observe_intervention(
    before: Grid,
    after: Grid,
    intervention: Mapping[str, Any],
    *,
    reward: float = 0.0,
    level_completed: bool = False,
    terminal: bool = False,
) -> dict[str, Any]:
    """Describe an observed state transition without assigning game semantics."""
    left, right = _rectangular(before), _rectangular(after)
    if not left or not right or len(left) != len(right) or len(left[0]) != len(right[0]):
        return {"transition_class": "shape_change", "changed_cells": None}
    rows, cols = len(left), len(left[0])
    changed = [
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if left[row][col] != right[row][col]
    ]
    bbox = None
    if changed:
        bbox = [
            min(point[0] for point in changed),
            min(point[1] for point in changed),
            max(point[0] for point in changed),
            max(point[1] for point in changed),
        ]
    changed_fraction = len(changed) / (rows * cols)
    bbox_fraction = 0.0
    if bbox:
        bbox_fraction = ((bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)) / (rows * cols)
    if not changed:
        scope = "none"
    elif changed_fraction >= 0.20 or bbox_fraction >= 0.50:
        scope = "global"
    elif intervention.get("row") is not None and intervention.get("col") is not None:
        row, col = int(intervention["row"]), int(intervention["col"])
        scope = (
            "target_local"
            if bbox and bbox[0] - 2 <= row <= bbox[2] + 2 and bbox[1] - 2 <= col <= bbox[3] + 2
            else "remote"
        )
    else:
        scope = "bounded"
    if not changed:
        magnitude = "zero"
    elif len(changed) <= 4:
        magnitude = "tiny"
    elif len(changed) <= 64:
        magnitude = "small"
    elif len(changed) <= 512:
        magnitude = "medium"
    else:
        magnitude = "large"
    motions = _motion_count(left, right) if changed else 0
    objective = "terminal" if terminal else "level" if level_completed else "reward" if reward else "none"
    roles = {str(value) for value in intervention.get("target_roles", [])}
    object_like = bool(roles & {"component", "component_center", "multicolor_island", "lattice_gap"})
    if objective != "none":
        causality = "objective"
    elif scope in {"target_local", "bounded"} and changed:
        causality = "likely_causal"
    elif not changed:
        causality = "latent_or_noop"
    else:
        causality = "uncertain_ambient"
    latent_selection = bool(
        str(intervention.get("action") or "").upper() == "MOUSE"
        and object_like
        and not terminal and not level_completed
        and (not changed or (scope == "remote" and len(changed) <= 4))
    )
    visible_followup = bool(
        str(intervention.get("action") or "").upper() == "MOUSE"
        and changed and scope in {"target_local", "bounded"}
        and not terminal and not level_completed
    )
    followup_reason = "latent_selection" if latent_selection else "visible_local" if visible_followup else ""
    transition_class = f"{scope}:{magnitude}:motion{int(bool(motions))}:objective_{objective}"
    return {
        "before_hash": grid_fingerprint(left),
        "after_hash": grid_fingerprint(right),
        "before_interior_hash": interior_fingerprint(left),
        "after_interior_hash": interior_fingerprint(right),
        "intervention_id": intervention.get("id"),
        "intervention_key": intervention_key(intervention),
        "intervention_class": intervention.get("intervention_class"),
        "target_id": intervention.get("target_id"),
        "changed_cells": len(changed),
        "changed_fraction": round(changed_fraction, 6),
        "bbox": bbox,
        "scope": scope,
        "motions": motions,
        "reward": float(reward),
        "level_completed": bool(level_completed),
        "terminal": bool(terminal),
        "transition_class": transition_class,
        "causality": causality,
        "latent_selection": latent_selection,
        "followup_reason": followup_reason,
        "followup_warranted": bool(latent_selection or visible_followup),
    }


def rank_interventions(
    frontier: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    *,
    pending_source: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Rank intervention novelty while discounting HUD-only revisits and reactive loops."""
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
    for intervention in list(frontier.get("interventions") or []):
        item = dict(intervention)
        intervention_class = str(item.get("intervention_class") or "")
        key = intervention_key(item)
        score = int(item.get("priority") or 0)
        score += 30 if class_attempts[intervention_class] == 0 else 0
        score += 6 if intervention_class in changed_classes else 0
        score += 120 * objective_classes[intervention_class]
        score -= 12 * no_change_classes[intervention_class]
        score -= 90 * exact_attempts[(intervention_class, key)]
        score -= 55 * interior_attempts[(intervention_class, key)]
        score -= min(24, 3 * max(0, class_attempts[intervention_class] - 1))
        if pending_source and item.get("action") == "MOUSE" and key != pending_key:
            score += 10
        item["intervention_key"] = key
        item["frontier_score"] = score
        ranked.append(item)
    return sorted(ranked, key=lambda item: (-int(item["frontier_score"]), str(item.get("action") or ""), str(item.get("intervention_key") or "")))


REGIME_EFFECT_RATE_THRESHOLD = 0.9838709677419355


def mouse_effect_rate(observations: Sequence[Mapping[str, Any]]) -> float:
    """Observed effect rate for prior MOUSE interventions in the rolling evidence window."""
    mouse = [
        observation for observation in observations
        if str(observation.get("intervention_class") or "").startswith("control:MOUSE")
    ]
    if not mouse:
        return 0.0
    effectful = sum(
        bool(observation.get("level_completed") or observation.get("terminal") or observation.get("reward"))
        or int(observation.get("changed_cells") or 0) > 0
        for observation in mouse
    )
    return effectful / len(mouse)

def rank_interventions_outcome_only(
    frontier: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rank by observed productive outcomes without novelty or revisit pressure."""
    changed_classes: set[str] = set()
    no_change_classes: Counter[str] = Counter()
    objective_classes: Counter[str] = Counter()
    for observation in observations:
        intervention_class = str(observation.get("intervention_class") or "")
        if observation.get("level_completed") or observation.get("terminal") or observation.get("reward"):
            objective_classes[intervention_class] += 1
        elif int(observation.get("changed_cells") or 0) > 0:
            changed_classes.add(intervention_class)
        else:
            no_change_classes[intervention_class] += 1

    ranked: list[dict[str, Any]] = []
    for intervention in list(frontier.get("interventions") or []):
        item = dict(intervention)
        intervention_class = str(item.get("intervention_class") or "")
        key = intervention_key(item)
        score = int(item.get("priority") or 0)
        score += 6 if intervention_class in changed_classes else 0
        score += 120 * objective_classes[intervention_class]
        score -= 12 * no_change_classes[intervention_class]
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


def rank_interventions_regime_gated(
    frontier: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    *,
    threshold: float = REGIME_EFFECT_RATE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Use outcome feedback only in a high-reliability MOUSE-effect regime."""
    rate = mouse_effect_rate(observations)
    if rate >= threshold:
        ranked = rank_interventions_outcome_only(frontier, observations)
        policy = "outcome_only"
    else:
        ranked = [dict(item) for item in list(frontier.get("interventions") or [])]
        policy = "static"
    for item in ranked:
        item["intervention_key"] = intervention_key(item)
        item["frontier_score"] = int(item.get("frontier_score", item.get("priority") or 0))
        item["regime_policy"] = policy
        item["regime_effect_rate"] = round(rate, 6)
    return ranked
