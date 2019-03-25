from heapq import heappush, heappop, nlargest, nsmallest
import numpy as np

DISTANCES = {
    'euclidian': lambda a, b: ((a - b) ** 2).sum(-1),
    'negative_dot': lambda a, b: - (a * b).sum(-1),
    'cosine_distance': lambda a, b: 1.0 - (a * b).sum(-1) / (a * a).sum(-1) ** 0.5 / (b * b).sum(-1) ** 0.5,
}


# Note: we use a * b and not einsum so that metric would work for both torch and numpy


class HNSW:
    """ Main class that handles approximate nearest neighbor lookup. Uses heap-based EFSearch. """

    def __init__(self, graph, ef=float('inf'), max_hops=float('inf'), max_dcs=float('inf'),
                 hierarchical=True, distance=DISTANCES['euclidian']):
        self.graph = graph
        self.hierarcical = hierarchical
        self.ef, self.max_hops, self.max_dcs = ef, max_hops, max_dcs
        if np.isinf(self.ef) and np.isinf(max_hops) and np.isinf(max_dcs):
            raise ValueError("Please specify a limit on max_hops, max_dcs or ef")
        assert distance in DISTANCES or callable(distance)
        self.get_distance = DISTANCES.get(distance, distance)

    def find_nearest(self, query, initial_vertex_id=None, **kwargs):
        """
        Performs nearest neighbor lookup and returns statistics.
        :param query: vector [vertex_size] to find nearest neighbor for
        :return: a dict with
            - 'best_vertex_id' - nearest neighbor vertex id
            - 'dcs' - number of distance computations
            - some other stats
        """
        vertex_id = self.get_initial_vertex_id(query, **kwargs) if initial_vertex_id is None else initial_vertex_id
        visited_ids = {vertex_id}  # a set of vertices already visited by graph walker

        topResults, candidateSet = [], []
        distance = self.get_distance(query, self.graph.vertices[vertex_id])
        heappush(topResults, (-distance, vertex_id))
        heappush(candidateSet, (distance, vertex_id))
        lowerBound = distance
        num_hops, dcs = 1, 1

        while len(candidateSet) > 0:
            dist, vertex_id = heappop(candidateSet)
            if np.isfinite(self.ef) and dist > lowerBound: break

            neighbor_ids = self.get_neighbors(vertex_id, visited_ids, **kwargs)
            if not len(neighbor_ids): continue

            visited_ids.update(neighbor_ids)

            distances = self.get_distance(query, self.graph.vertices[neighbor_ids])
            for distance, neighbor_id in zip(distances, neighbor_ids):
                if distance < lowerBound or len(topResults) < self.ef:
                    heappush(candidateSet, (distance, neighbor_id))
                    heappush(topResults, (-distance, neighbor_id))

                    if len(topResults) > self.ef:
                        heappop(topResults)

                    lowerBound = -nsmallest(1, topResults)[0][0]

                dcs += 1
                if dcs >= self.max_dcs: break

            num_hops += 1
            if num_hops >= self.max_hops: break
            if dcs >= self.max_dcs: break

        best_vertex_id = nlargest(1, topResults)[0][1]
        return dict(
            best_vertex_id=best_vertex_id, dcs=dcs, num_hops=num_hops,
            top_results_heap=topResults, visited_ids=visited_ids
        )

    def get_initial_vertex_id(self, query=None, **kwargs):
        if not self.hierarcical:
            return self.graph.initial_vertex_id
        assert query is not None

        vertex_id = self.graph.initial_vertex_id
        curdist = self.get_distance(query, self.graph.vertices[vertex_id])

        for level in range(self.graph.max_level)[::-1]:
            changed = True
            while changed:
                changed = False
                edges = list(self.graph.level_edges[vertex_id][level])
                if len(edges) == 0:
                    break

                distances = self.get_distance(query, self.graph.vertices[edges])
                for edge, dist in zip(edges, distances):
                    if dist < curdist:
                        curdist = dist
                        vertex_id = edge
                        changed = True
        return vertex_id

    def get_neighbors(self, vertex_id, visited_ids, **kwargs):
        """ :return: a list of neighbor ids available from given vector_id. """
        neighbors = [edge for edge in self.graph.edges[vertex_id]
                     if edge not in visited_ids]
        return neighbors


class WalkerHNSW(HNSW):
    def __init__(self, *args, distance=DISTANCES['euclidian'], distance_for_routing=None,
                 top_vertices_for_verification=1, distance_for_verification=None,
                 **kwargs):
        """
        :param max_hops: maximum iterations of main loop before forced stop
        :param max_dcs: maximum distance comptations before forced stop
        :param distance_for_routing: distance used when performing approximate search
        :param distance_for_verification: distance used when selecting best vertices among top-k
        :param top_vertices_for_verification: selects best answer among this many top edges
        :param edge_level: if True, every edge is treated unique.
            if False, edges corresponding to the same destination are condidered equivalent
        """
        super().__init__(*args, **kwargs, distance=distance)
        distance = distance if callable(distance) else DISTANCES[distance]

        def _make_distance(d):
            d = DISTANCES.get(d, d) or distance
            assert callable(d), "could not interpret {} as distance".format(d)
            return d

        self.distance_for_routing = _make_distance(distance_for_routing)
        self.distance_for_verification = _make_distance(distance_for_verification)
        self.top_vertices_for_verification = top_vertices_for_verification

    def find_nearest(self, query, *, agent, state=None, **kwargs):
        """
        Performs nearest neighbor lookup and returns statistics.
        :param query: vector [vertex_size] to find nearest neighbor for
        :type agent: lib.walker_agent.BaseWalkerAgent
        :return: nearest neighbor vertex id
        """
        if state is None:
            state = agent.prepare_state(self.graph, **kwargs)

        initial_vertex_id = self.get_initial_vertex_id(query, **kwargs)
        query_vector = agent.get_query_vectors(query[None], state=state, **kwargs)[0]  # [vector_size]

        assert isinstance(initial_vertex_id, (type(None), int))

        # Below: initial edge (loop at initial vertex id)
        # a fake edge representing the starting point of search algo
        initial_vertex_vector = agent.get_vertex_vectors([initial_vertex_id], state=state, **kwargs)[0]
        initial_distance = self.distance_for_routing(query_vector, initial_vertex_vector).item()

        candidates = []  # heap of vertices from smallest predicted distance to largest
        heappush(candidates, (initial_distance, initial_vertex_id))
        top_results = []  # heap of top-ef vertices from largest predicted distance to smallest. Used for pruning
        heappush(top_results, (-initial_distance, initial_vertex_id))
        visited_ids = {initial_vertex_id}  # a set of vertices already visited by graph walker

        neg_lower_bound_distance, lower_bound_vertex_id = nsmallest(1, top_results)[0]
        lower_bound_distance = -neg_lower_bound_distance
        dcs, num_hops = 1, 1

        while len(candidates) != 0:

            # 1. pop edge according to graph walker
            estimated_distance, vertex_id = heappop(candidates)
            if np.isfinite(self.ef) and estimated_distance > lower_bound_distance: break

            # 2. gather all next vertices
            neighbor_ids = self.get_neighbors(vertex_id, visited_ids, **kwargs)

            # update visited ids
            visited_ids.update(neighbor_ids)

            # 3. compute distances and add all neighbors to candidates
            if len(neighbor_ids) > 0:
                neighbor_vectors = agent.get_vertex_vectors(neighbor_ids, state=state, **kwargs)
                distances = self.distance_for_routing(query_vector[None], neighbor_vectors).data.cpu().numpy()
            else:
                distances = []

            for distance, neighbor_id in zip(map(float, distances), neighbor_ids):

                if distance < lower_bound_distance or len(top_results) < self.ef:
                    heappush(candidates, (distance, neighbor_id))
                    heappush(top_results, (-distance, neighbor_id))

                    if len(top_results) > self.ef:
                        heappop(top_results)

                    neg_lower_bound_distance, lower_bound_vertex_id = nsmallest(1, top_results)[0]
                    lower_bound_distance = -neg_lower_bound_distance
                else:
                    pass  # pruned by lower bound

                # early stopping by dcs
                dcs += 1
                if dcs >= self.max_dcs: break

            # early stopping by dcs
            num_hops += 1
            if num_hops >= self.max_hops: break
            if dcs >= self.max_dcs: break

        # select best vertex
        verification_top = nlargest(self.top_vertices_for_verification, top_results)
        vertices_for_verification = [chosen_vertex_id for _neg_distance, chosen_vertex_id in verification_top]
        vertices_for_verification = list(set(vertices_for_verification))

        if len(vertices_for_verification) == 1:
            _neg_distance, best_vertex_id = verification_top[0]
            verification_dcs = 0
        else:
            # manually select best from top-k vertices acc. to actual distance
            verification_distances = self.distance_for_verification(query, self.graph.vertices[vertices_for_verification])
            # ^-- [len(verification_top)]
            best_vertex_id = vertices_for_verification[np.argmin(verification_distances.data.cpu().numpy())]
            verification_dcs = len(verification_distances)

        compression_rate = float(query.shape[-1]) / float(query_vector.shape[-1])

        return dict(
            best_vertex_id=best_vertex_id,  # <-- your answer :)
            dcs=dcs / compression_rate + len(verification_top), num_hops=num_hops,
            top_results_heap=top_results, visited_ids=visited_ids, verification_ids=vertices_for_verification,
            compressed_dcs=dcs + len(verification_top) * compression_rate, compression_rate=compression_rate,
            routing_dcs=dcs, verification_dcs=verification_dcs,
        )
