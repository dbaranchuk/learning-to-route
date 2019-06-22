import numpy as np
import multiprocessing
from concurrent.futures import Future
from threading import Thread
from .wrapper import bfs


class ParallelBFS:
    def __init__(self, *, graph_edges=None, inverse_edges=None, n_jobs=None):
        """
        A wrapper for c++ code that computes optimal paths in parallel. Used in training batch generator.
        Please provide either graph_edges or inverse_edges
        :param graph_edges: a dict{vertex_id -> [successor ids]}
        :param inverse_edges: a dict{vertex_id -> [predecessor ids]}
        :param n_jobs: default number of threads used to compute several queries in parallel
        """
        assert (graph_edges is None) != (inverse_edges is None), "please provide either edges or inverse edges"
        if inverse_edges is None:
            assert isinstance(graph_edges, dict)
            inverse_edges = {vid: list() for vid in graph_edges.keys()}
            for from_i, to_ix in graph_edges.items():
                for to_i in to_ix:
                    inverse_edges[to_i].append(from_i)
        self.inverse_edges = self._check_edges(inverse_edges)
        self.n_jobs = self._check_n_jobs(n_jobs)

    def __call__(self, target_vertex_ids, visited_ids, n_jobs=None):
        """
        :param target_vertex_ids: array[int32, num_queries] of target vertex indices
        :param visited_ids: list of lists, for each target vertex, a list of
            vertex ids from which you want to compute distance to that target vertex
        :param n_jobs: number of parallel threads used for search
        :param use_arraydicts: if True, returned dicts are ArrayDict (faster for large margin)
        :return: a list[dict{v -> topologic distance to gt}] for gt in target_vertex_ids
        """
        n_jobs = self.n_jobs if n_jobs is None else self._check_n_jobs(n_jobs)
        target_vertex_ids = np.asarray(target_vertex_ids, dtype=np.int32)
        assert np.ndim(target_vertex_ids) == 1
        num_queries = len(target_vertex_ids)
        max_visited_ids = max(map(len, visited_ids))

        visited_ids_matrix = np.full([num_queries, max_visited_ids], -1, dtype=np.int32)
        for i, vertices in enumerate(visited_ids):
            visited_ids_matrix[i, :len(vertices)] = vertices

        # v-- buffer for outputs, matrix of shape [n_queries, max_visited_ids], should be padded with -1
        distances_matrix = np.full([num_queries, max_visited_ids], -1, dtype=np.int32)

        bfs(self.inverse_edges, target_vertex_ids, distances_matrix, visited_ids_matrix, n_jobs)

        vertex_id_to_distance = [
            dict(zip(visited_ids_i, distances_i))
            for visited_ids_i, distances_i in zip(visited_ids_matrix, distances_matrix)
        ]
        for row in vertex_id_to_distance:
            row.pop(-1, None)

        return vertex_id_to_distance

    def compute_paths_async(self, initial_vertex_ids, target_vertex_ids, margin=None, n_jobs=None):
        """ Launches parallel BFS in a background thread. returns Future for the result """
        future_output = Future()

        def thread_method():
            result = self(initial_vertex_ids, target_vertex_ids, margin=margin, n_jobs=n_jobs)
            future_output.set_result(result)

        Thread(target=thread_method).start()
        return future_output

    @staticmethod
    def _check_edges(graph_edges):
        if isinstance(graph_edges, dict):
            assert max(graph_edges.keys()) + 1 == len(graph_edges)
            num_vertices = len(graph_edges)
            max_edges = max(map(len, graph_edges.values()))
            edges_table = np.full([num_vertices, max_edges], -1, dtype=np.int32)
            for from_i, to_ix in graph_edges.items():
                edges_table[from_i, :len(to_ix)] = to_ix
            return edges_table
        elif isinstance(graph_edges, np.ndarray):
            return graph_edges
        else:
            raise NotImplementedError("unsupported edges type: " + repr(type(graph_edges)))

    @staticmethod
    def _check_n_jobs(n_jobs):
        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count()
        if n_jobs < 0:
            n_jobs = multiprocessing.cpu_count() + 1 - n_jobs
        assert n_jobs > 0
        return n_jobs
