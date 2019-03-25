import os
from multiprocessing.pool import Pool
from heapq import heappush, heappop

import diskcache
from struct import pack, unpack
from tqdm import tqdm
from itertools import count


class Dijkstra:
    def __init__(self, graph, minimize_dcs=False):
        self.graph = graph
        self.minimize_dcs = minimize_dcs
        self.opposite_edges = {i: [] for i in range(len(graph.vertices))}
        for from_i, to_ix in graph.edges.items():
            for to_i in to_ix:
                self.opposite_edges[to_i].append(from_i)

    def __call__(self, initial_id, target_id, margin=0, prune_inf=True):
        """
        :param target_id: reference answer vertex id
        :param initial_id: vertex to start from
        :param minimize_dcs: if True, minimizes total distance computations along optimal path
            if False (default), minimizes the number of hops
        :param margin: considers paths this much longer than optimal path
        :param prune_inf: if True, removes all vertices with distance = infinity
        :returns: id_to_dist, dictionary { vertex_id : distance to gt }
        """
        if initial_id is None:
            initial_id = self.graph.initial_vertex_id
        dist_to_target = _compute_distances_to_target(initial_id, target_id, self.graph.edges, self.opposite_edges,
                                                      minimize_dcs=self.minimize_dcs, margin=margin)
        dist_to_initial = _compute_distances_to_target(target_id, initial_id, self.opposite_edges, self.graph.edges,
                                                       minimize_dcs=self.minimize_dcs, margin=margin)

        optimal_distance = dist_to_target[initial_id]
        return {
            vertex_id: dist_to_target[vertex_id] for vertex_id in dist_to_target
            if (dist_to_initial.get(vertex_id, float('inf')) + dist_to_target.get(vertex_id, float('inf'))
                <= optimal_distance + margin)
        }


def _compute_distances_to_target(initial_id, target_id, edges, opposite_edges,
                                 minimize_dcs=False, margin=0):
    visited_ids = set()
    id_queue = []
    heappush(id_queue, (0, target_id))
    id_to_dist = dict()
    for vertex in range(len(edges)):
        id_to_dist[vertex] = float('inf')
    id_to_dist[target_id] = 0

    while (len(id_queue) > 0):
        distance, next_id = heappop(id_queue)
        if distance > id_to_dist[initial_id] + margin:
            break

        edge_cost = len(edges[next_id]) if minimize_dcs else 1

        for neighbor in opposite_edges[next_id]:
            if (neighbor in visited_ids):
                continue
            if (id_to_dist[neighbor] > id_to_dist[next_id] + edge_cost):
                id_to_dist[neighbor] = id_to_dist[next_id] + edge_cost
                heappush(id_queue, (id_to_dist[neighbor], neighbor))

        visited_ids.add(next_id)

    return id_to_dist


class PathCache:
    def __init__(self, path, finder, reset_cache=False, timeout=10**10, size_limit=10**18, **kwargs):
        """ Stores pre-computed dijkstra distance dictionaries on disk with quick access """
        if reset_cache:
            os.system("rm -rf {}".format(path))
        os.system("mkdir -p {}".format(path))

        self.cache = diskcache.Cache(path, timeout=timeout, size_limit=size_limit)
        self.dijkstra = finder
        self.kwargs = kwargs

    def export_from_bin(self, bin_file):
        with open(bin_file, "rb") as f:
            for idx in tqdm(count()):
                header = f.read(12)
                if not len(header):
                    break
                from_id, to_id, size = unpack("III", header)
                data = unpack(size * "I", f.read(4 * size))

                vertex_id_to_dist = {data[i]: data[i + 1] for i in range(0, size, 2)}
                self[(from_id, to_id)] = vertex_id_to_dist

    def __call__(self, initial_id, target_id, recache=False):
        if recache or (initial_id, target_id) not in self.cache:
            path = self.dijkstra(initial_id, target_id, **self.kwargs)
            self.cache[(int(initial_id), int(target_id))] = path
        return self[initial_id, target_id]

    def __getitem__(self, initial_and_target_tuple):
        from_id, to_id = map(int, initial_and_target_tuple)
        try:
            return self.cache[from_id, to_id]
        except:
            self.cache[from_id, to_id] = self.dijkstra(from_id, to_id, **self.kwargs)
            return self.cache[from_id, to_id]

    def __setitem__(self, initial_and_target_tuple, vertex_id_to_dist):
        from_id, to_id = map(int, initial_and_target_tuple)
        self.cache[from_id, to_id] = vertex_id_to_dist

    def _preprocess(self, initial_and_target):
        """ precomputes path, saves to cache, returns key """
        initial_id, target_id = map(int, initial_and_target)
        return ((initial_id, target_id), self.dijkstra(initial_id, target_id, **self.kwargs))

    def prepare_paths(self, initial_vertices, target_vertices, batch_size=512, pool=None):
        assert len(initial_vertices) == len(target_vertices)
        tasks = [
            zip(initial_vertices[batch_start: batch_start + batch_size],
                target_vertices[batch_start: batch_start + batch_size])
            for batch_start in range(0, len(initial_vertices), batch_size)
        ]
        if not isinstance(pool, Pool):
            pool = Pool(pool)

        for task in tqdm(tasks):
            batch_keys_and_paths = pool.map(self._preprocess, task)
            for key, path in batch_keys_and_paths:
                self[key] = path
