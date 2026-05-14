"""
network_graph.py — Trusted Edge Pool Builder

Algorithm: O(V+E) BFS using lane-link adjacency.
Forward BFS from seed → edges reachable FROM seed.
Backward BFS from seed → edges that CAN REACH seed.
Trusted pool = intersection (strongly-connected component containing seed).
"""

import sys
from collections import deque, defaultdict
import traci

TRUSTED_EDGES: list = []
TRUSTED_SET:   set  = set()


def _build_lane_adjacency(all_edges: list) -> tuple:
    forward  = defaultdict(set)
    backward = defaultdict(set)
    edge_set = set(all_edges)

    for edge in all_edges:
        try:
            num_lanes = traci.edge.getLaneNumber(edge)
        except traci.TraCIException:
            continue
        for i in range(num_lanes):
            _follow_links(f"{edge}_{i}", edge, forward, backward, edge_set, depth=0)

    return forward, backward


def _follow_links(lane_id, src_edge, forward, backward, edge_set, depth):
    if depth > 4:
        return
    try:
        links = traci.lane.getLinks(lane_id)
    except traci.TraCIException:
        return
    for link in links:
        next_lane = link[0]
        if not next_lane:
            continue
        edge_candidate = next_lane.rsplit("_", 1)[0] if "_" in next_lane else next_lane
        if edge_candidate.startswith(":"):
            _follow_links(next_lane, src_edge, forward, backward, edge_set, depth + 1)
        elif edge_candidate in edge_set and edge_candidate != src_edge:
            forward[src_edge].add(edge_candidate)
            backward[edge_candidate].add(src_edge)


def _bfs(start: str, adj: dict, universe: set) -> set:
    visited = {start}
    queue   = deque([start])
    while queue:
        node = queue.popleft()
        for nbr in adj.get(node, ()):
            if nbr in universe and nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)
    return visited


def build_trusted_pool() -> None:
    global TRUSTED_EDGES, TRUSTED_SET

    print("🔍 Building trusted edge pool (BFS, O(V+E))...")
    sys.stdout.flush()

    all_edges = []
    for e in traci.edge.getIDList():
        if e.startswith(":"):
            continue
        try:
            if traci.edge.getLaneNumber(e) > 0:
                all_edges.append(e)
        except traci.TraCIException:
            pass

    print(f"   Candidate edges  : {len(all_edges)}")

    forward, backward = _build_lane_adjacency(all_edges)

    seed = max(all_edges,
               key=lambda e: len(forward.get(e, set())),
               default=all_edges[0])
    print(f"   Seed edge        : {seed} (out-degree {len(forward[seed])})")

    fwd_reach = _bfs(seed, forward,  set(all_edges))
    print(f"   Forward reach    : {len(fwd_reach)}")

    bwd_reach = _bfs(seed, backward, fwd_reach)
    print(f"   Backward reach   : {len(bwd_reach)}")

    trusted      = fwd_reach & bwd_reach
    TRUSTED_EDGES = list(trusted)
    TRUSTED_SET   = trusted

    removed = len(all_edges) - len(TRUSTED_EDGES)
    print(f"✅ Trusted pool     : {len(TRUSTED_EDGES)} edges "
          f"({removed} isolated/dead-end edges removed)\n")


def get_trusted_edges() -> list: return TRUSTED_EDGES
def get_trusted_set()   -> set:  return TRUSTED_SET
