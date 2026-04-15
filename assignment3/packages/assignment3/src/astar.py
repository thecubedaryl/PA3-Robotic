#!/usr/bin/env python3
"""
A* path planning from scratch (no path-planning libraries).
Grid graph N0..N15 with coordinates and edge costs as specified for PA3.
"""
from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Set, Tuple

# Node coordinates (x, y) — indices match node id
COORDINATES: Dict[int, Tuple[float, float]] = {
    0: (0.0, 0.0),
    1: (1.0, 0.0),
    2: (2.0, 0.0),
    3: (3.0, 0.0),
    4: (0.0, 1.0),
    5: (1.0, 1.0),
    6: (2.0, 1.0),
    7: (3.0, 1.0),
    8: (0.0, 2.0),
    9: (1.0, 2.0),
    10: (2.0, 2.0),
    11: (3.0, 2.0),
    12: (0.0, 3.0),
    13: (1.0, 3.0),
    14: (2.0, 3.0),
    15: (3.0, 3.0),
}

# Undirected graph: neighbor -> edge cost (exactly as in assignment)
GRAPH: Dict[int, List[Tuple[int, float]]] = {
    0: [(1, 2.0), (4, 2.0)],
    1: [(0, 2.0), (2, 1.0), (5, 1.5)],
    2: [(1, 1.0), (3, 1.0), (6, 1.5)],
    3: [(2, 1.0), (7, 1.0)],
    4: [(0, 2.0), (5, 2.0), (8, 1.5)],
    5: [(1, 1.5), (4, 2.0), (9, 1.5)],
    6: [(2, 1.5), (7, 0.5), (10, 4.0)],
    7: [(3, 1.0), (6, 0.5), (11, 1.5)],
    8: [(4, 1.5), (9, 1.5), (12, 2.0)],
    9: [(5, 1.5), (8, 1.5), (13, 1.5)],
    10: [(6, 4.0), (11, 1.0), (14, 1.5)],
    11: [(7, 1.5), (10, 1.0), (15, 1.0)],
    12: [(8, 2.0), (13, 1.5)],
    13: [(9, 1.5), (12, 1.5), (14, 2.0)],
    14: [(10, 1.5), (13, 2.0), (15, 1.5)],
    15: [(11, 1.0), (14, 1.5)],
}


def euclidean_heuristic(node: int, goal: int) -> float:
    x1, y1 = COORDINATES[node]
    x2, y2 = COORDINATES[goal]
    return math.hypot(x2 - x1, y2 - y1)


def astar_search(start: int, goal: int) -> Tuple[Optional[List[int]], float]:
    """
    A* with:
    - OPEN: min-heap ordered by f, tie-break lower h(n)
    - CLOSED: set of expanded nodes
    - Each reached node tracks g(n); f = g + h; parent for reconstruction
    """
    if start not in COORDINATES or goal not in COORDINATES:
        return None, float("nan")

    def h(n: int) -> float:
        return euclidean_heuristic(n, goal)

    # OPEN: min-heap of (f, h, g, node); tie-break: lower h first when f is equal
    OPEN: List[Tuple[float, float, float, int]] = []
    CLOSED: Set[int] = set()
    g_score: Dict[int, float] = {start: 0.0}
    parent: Dict[int, int] = {}

    h0 = h(start)
    heapq.heappush(OPEN, (h0, h0, 0.0, start))

    while OPEN:
        _f, _h, g, current = heapq.heappop(OPEN)
        if current in CLOSED:
            continue
        if g > g_score[current] + 1e-9:
            continue

        CLOSED.add(current)

        if current == goal:
            path: List[int] = []
            cur: Optional[int] = goal
            while cur is not None:
                path.append(cur)
                if cur == start:
                    break
                cur = parent.get(cur)
            path.reverse()
            if not path or path[0] != start or path[-1] != goal:
                return None, float("nan")
            return path, g_score[goal]

        for nbr, step_cost in GRAPH[current]:
            if nbr in CLOSED:
                continue
            tentative_g = g + step_cost
            if tentative_g < g_score.get(nbr, float("inf")) - 1e-9:
                g_score[nbr] = tentative_g
                parent[nbr] = current
                hn = h(nbr)
                fn = tentative_g + hn
                heapq.heappush(OPEN, (fn, hn, tentative_g, nbr))

    return None, float("nan")


def format_path(path: List[int]) -> str:
    return " → ".join("N%d" % n for n in path)


def _report(start: int, goal: int) -> None:
    path, cost = astar_search(start, goal)
    if path is None:
        print("No path found from N%d to N%d." % (start, goal))
        return
    print("Path sequence: %s" % format_path(path))
    print("Total cost: %.12g" % cost)


if __name__ == "__main__":
    _report(0, 15)
