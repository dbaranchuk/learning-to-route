import time
import torch
import joblib
import numpy as np
from heapq import heappush, heappop, nsmallest, nlargest
from tensorboardX import SummaryWriter
from datetime import datetime

import lib
from lib.utils import global_variables, require_variables, iterate_minibatches
from lib.nn_utils import get_device_of


class SupervisedWalkerTrainer:
    def __init__(self, agent, hnsw, oracle, writer=None,
                 Optimizer=lambda params: torch.optim.Adam(params, lr=1e-4, amsgrad=True),
                 device=None, **learning_rate_opts):
        """
        A class that handles agent training
        :type agent: lib.walker_agent.BaseWalkerAgent
        :type hnsw: lib.hnsw.WalkerHNSW
        :type oracle: lib.ParallelBFS
        :param device: device to run computations on (string)
        """
        self.agent, self.hnsw, self.oracle = agent, hnsw, oracle
        self.opt = Optimizer(agent.parameters())
        self.device = device or get_device_of(agent)
        self.learning_rate_opts = learning_rate_opts
        self.step = 0
        default_path = './runs/untitled_' + str(datetime.now()).split('.')[0].replace(' ', '_')
        self.writer = writer or SummaryWriter(writer or default_path)

    def train_on_batch(self, batch_records, prefix='train/', **kwargs):
        """ Trains on records sampled from self.generate_training_batches """
        start_time = time.time()
        learning_rate = self.update_learning_rate(self.step, **self.learning_rate_opts)
        batch_metrics = self.compute_metrics_on_batch(batch_records, **kwargs)
        batch_metrics['loss'].backward()
        self.opt.step()
        self.opt.zero_grad()

        batch_metrics['learning_rate'] = learning_rate
        batch_metrics['records_per_batch'] = len(batch_records)
        batch_metrics['step_time'] = time.time() - start_time
        for key, value in batch_metrics.items():
            self.writer.add_scalar(prefix + key, value, global_step=self.step)
        self.step += 1
        return batch_metrics

    def update_learning_rate(self, t, learning_rate_base=1e-3, warmup_steps=5000,
                             decay_rate=1./3, learning_rate_min=1e-5, **kwargs):
        """ Learning rate with linear warmup and exponential decay """
        lr = learning_rate_base * np.minimum(
            (t + 1.0) / warmup_steps,
            np.exp(decay_rate * ((warmup_steps - t - 1.0) / warmup_steps)),
        )
        lr = np.maximum(lr, learning_rate_min)
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr
        return lr

    def compute_metrics_on_batch(self, batch_records, **kwargs):
        # step 1: compute vectors for queries
        state = self.agent.prepare_state(self.hnsw.graph, device=self.device, **kwargs)
        batch_queries = torch.tensor([rec['query'] for rec in batch_records], device=self.device)
        batch_query_vectors = self.agent.get_query_vectors(batch_queries, state=state,
                                                           device=self.device, **kwargs)

        # step 2: precompute all edge vectors, deduplicate for efficiency
        all_vertices = set()
        vertices_per_record = []
        for rec in batch_records:
            rec_edges = set()
            for row in rec['objectives']:
                rec_edges.update(row['positive_ids'])
                rec_edges.update(row['negative_ids'])
            all_vertices.update(rec_edges)
            vertices_per_record.append(rec_edges)

        all_vertices = list(all_vertices)
        vertex_to_ix = {edge: i for i, edge in enumerate(all_vertices)}

        batch_vertex_vectors = self.agent.get_vertex_vectors(
            all_vertices, state=state, device=self.device, **kwargs)

        # compute all distances
        query_ix, vertex_ix, is_numerator, row_ix, col_ix, row_weights = [], [], [], [], [], []
        row_index = 0
        for i, rec in enumerate(batch_records):
            for row in rec['objectives']:
                num_vertices = (len(row['positive_ids']) + len(row['negative_ids']))
                query_ix.extend([i] * num_vertices)
                vertex_ix.extend([vertex_to_ix[vertex_id] for vertex_id in row['positive_ids'] + row['negative_ids']])
                is_numerator.extend([1] * len(row['positive_ids']) + [0] * len(row['negative_ids']))
                row_ix.extend([row_index] * num_vertices)
                col_ix.extend(range(num_vertices))
                row_weights.append(row.get('weight', 1.0))
                row_index += 1

        distances = self.hnsw.distance_for_routing(batch_query_vectors[query_ix], batch_vertex_vectors[vertex_ix])
        is_numerator = torch.tensor(is_numerator, dtype=torch.uint8, device=distances.device)
        row_ix = torch.tensor(row_ix, dtype=torch.int64, device=distances.device)
        col_ix = torch.tensor(col_ix, dtype=torch.int64, device=distances.device)
        row_weights = torch.tensor(row_weights, dtype=torch.float32, device=distances.device)

        logits = -distances

        # construct two matrices, both of shape [num_training_instances, num_vertices_per_instance]
        # first matrix contains all logits, padded by -inf
        all_logits_matrix = torch.full([row_index, col_ix.max() + 1], -1e9, device=logits.device)
        all_logits_matrix[row_ix, col_ix] = logits

        # second matrix only contains reference logits only
        ref_logits_matrix = torch.full([row_index, col_ix.max() + 1], -1e9, device=logits.device)
        ref_logits_matrix[row_ix, col_ix] = torch.where(is_numerator, logits, torch.full_like(logits, -1e9))

        logp_any_ref = torch.logsumexp(ref_logits_matrix, dim=1) \
                       - torch.logsumexp(all_logits_matrix, dim=1)

        xent = -torch.sum(logp_any_ref * row_weights) / torch.sum(row_weights)
        acc = torch.sum(torch.ge(torch.exp(logp_any_ref), 0.5).to(dtype=torch.float32)
                        * row_weights) / torch.sum(row_weights)
        return dict(loss=xent, xent=xent, acc=acc)

    def compute_dev_metrics_on_batch(self, batch_queries, batch_gt, prefix='dev/', **kwargs):
        """ Computes recall, dcs, ..., etc."""
        kwargs['device'] = kwargs.pop('device', self.device)
        with torch.no_grad():
            state = self.agent.prepare_state(self.hnsw.graph, **kwargs)
            results = [
                self.hnsw.find_nearest(q, agent=self.agent, state=state, **kwargs)
                for q in batch_queries
            ]

        recall = np.mean([res['best_vertex_id'] in gt for res, gt in zip(results, batch_gt)])
        dcs = np.mean([res['dcs'] for res in results])
        num_hops = np.mean([res['num_hops'] for res in results])
        batch_metrics = dict(recall=recall, dcs=dcs, num_hops=num_hops)

        if prefix is not None:
            for key, value in batch_metrics.items():
                self.writer.add_scalar(prefix + key, value, global_step=self.step)
        return batch_metrics

    def generate_training_batches(
            self, queries, ground_truth, initial_vertices=None,
            queries_per_chunk=None, batch_size_max=4096, n_jobs=-1,
            cycle=True, shuffle=True, verbose=True, **kwargs):
        """ Generates minibatches of records used for train_on_batch """
        queries_per_chunk = queries_per_chunk or len(queries)
        if initial_vertices is None:
            if verbose:
                print(end='Precomputing initial vertices... ', flush=True)
            initial_vertices = torch.tensor([
                self.hnsw.get_initial_vertex_id(q, **kwargs) for q in queries
            ], dtype=ground_truth.dtype, device=ground_truth.device)
            if verbose:
                print('Done!', flush=True)
        # batch buffers
        batch_records, total_size = [], 0

        for chunk_queries, chunk_initial_vertices, chunk_gt in iterate_minibatches(
                queries, initial_vertices, ground_truth,
                batch_size=queries_per_chunk, cycle=cycle, shuffle=shuffle
        ):
            if verbose:
                print(end="\nGenerating new batch of trajectories... ", flush=True)

            records = self.generate_records(queries=chunk_queries,
                                            initial_vertices=chunk_initial_vertices.data.numpy(),
                                            ground_truth=chunk_gt.data.numpy()[:, 0],
                                            n_jobs=n_jobs, device=self.device, **kwargs)
            if verbose:
                print("Done!", flush=True)

            for rec in records:
                if len(batch_records) > 0 and total_size + len(rec['objectives']) > batch_size_max:
                    yield batch_records
                    batch_records, total_size = [], 0

                batch_records.append(rec)
                total_size += len(rec['objectives'])


    def generate_records(self, *, queries, initial_vertices, ground_truth,
                         batch_size=None, n_jobs=-1, timeout=None, **kwargs):
        """ Generate a batch of training records by pre-computing graph vertices and sampling trajectories """
        assert np.ndim(initial_vertices) == 1 and np.ndim(ground_truth) == 1
        agent, hnsw, oracle = self.agent, self.hnsw, self.oracle
        query_vectors, vertex_id_to_vectors = prepare_vectors(agent, hnsw.graph, queries,
                                                              batch_size=batch_size, **kwargs)

        with global_variables(hnsw=hnsw, vertex_id_to_vectors=vertex_id_to_vectors, oracle=oracle):
            # ^-- we implicitly pass read-only variables to _sample_job cuz it's faster this way
            # 1. sample trajectories
            make_sample_job = joblib.delayed(_sample_job)
            samples_jobs = (make_sample_job(*args, **kwargs)
                        for args in zip(query_vectors, initial_vertices, ground_truth))
            trajectories = joblib.Parallel(n_jobs=n_jobs, timeout=timeout, backend='multiprocessing')(samples_jobs)

            # 2. compute true number of hops from each visited vertex to ground truth
            visited_ids = []
            for record in trajectories:
                all_vertices = list(record['visited_ids'] | {
                    record['initial_vertex_id'], record['ground_truth_id']})
                visited_ids.append(all_vertices)

            oracle_distances = oracle(ground_truth, visited_ids, n_jobs=n_jobs)
            # ^-- list of [{vertex id -> dist}]
            # 3. compute objectives along sampled :trajectories: using :oracle_distances:
            objectives = [_supervision_job(*args)
                          for args in zip(queries.data.numpy(), trajectories, oracle_distances)]

            # Note: using joblib twice sometimes leads to memory leak. So we comment it for stability.
            # make_supervision_job = joblib.delayed(_supervision_job)
            # supervision_jobs = (make_supervision_job(*args, **kwargs)
            #         for args in zip(queries.data.numpy(), trajectories, oracle_distances))
            # objectives = parallel_pool(supervision_jobs)
            return objectives


def prepare_vectors(agent, graph, queries, batch_size=None, **kwargs):
    """ Pre-computes all query and edge vectors to sample trajectories """
    with torch.no_grad():
        # pre-compute query vectors
        batch_size = batch_size or len(queries)
        state = agent.prepare_state(graph)
        query_vectors = []
        for batch_start in range(0, len(queries), batch_size):
            batch_queries = queries[batch_start: batch_start + batch_size]
            batch_query_vectors = agent.get_query_vectors(batch_queries, state=state,
                                                          **kwargs).data.cpu().numpy()
            query_vectors.append(batch_query_vectors)

        query_vectors = np.vstack(query_vectors)

        vertex_id_to_vectors = lib.walker_agent.PrecomputedWalkerAgent(agent).prepare_state(
            graph, state=state, batch_size=batch_size, **kwargs).vertex_vectors
    return query_vectors, vertex_id_to_vectors


def _sample_job(query_vec, initial_vertex_id, ground_truth_id, **kwargs):
    """ A wrapper that samples one trajectory, used with joblib """
    hnsw, vertex_id_to_vectors, oracle = require_variables('hnsw', 'vertex_id_to_vectors', 'oracle')
    initial_vertex_id, ground_truth_id = map(int, (initial_vertex_id, ground_truth_id))
    return sample_agent_trajectory(
        hnsw, query_vec, vertex_id_to_vectors,
        initial_vertex_id=initial_vertex_id, ground_truth_id=ground_truth_id, **kwargs)


def _supervision_job(query, recorded_trajectory, vertex_id_to_dist, **kwargs):
    """ A wrapper that computes all objectives along trajectory, used with joblib """
    hnsw, vertex_id_to_vectors, oracle = require_variables('hnsw', 'vertex_id_to_vectors', 'oracle')
    return compute_expert_supervision(hnsw, query, recorded_trajectory, vertex_id_to_dist, **kwargs)


def sample_agent_trajectory(hnsw, query_vector, vertex_id_to_vectors, *,
                            initial_vertex_id, ground_truth_id, terminate_on_gt=False,
                            sampling_temperature=1.0, eps=1e-6, **kwargs):
    """ Samples EF-search trajectory using pre-computed vectors
    :type hnsw: lib.hnsw.WalkerHNSW
    :param query_vector: query representation from agent.get_query_vector
    :param vertex_id_to_vectors: numpy array [vertices_size x vector_size]
    :param initial_vertex_id: vertex id to start from
    :param ground_truth_id: target that serch *should* find
    :param terminate_on_gt: if True, ends search immediately after visiting target vertex
    :param sampling_temperature: coefficient for gumbel noise added to distances.
        temperature=0 -> move exactly as during test, temperature > 0 -> more random moves
        higher temperature encourages exploration
    """
    assert isinstance(initial_vertex_id, (type(None), int))

    initial_distance = hnsw.distance_for_routing(query_vector, vertex_id_to_vectors[initial_vertex_id])
    if sampling_temperature != 0:
        noise = -np.log(-np.log(np.random.uniform(0.0, 1.0) + eps) + eps)
        initial_distance -= noise * sampling_temperature

    # training history: a list of training examples, examples are converted to loss after they are collected
    training_history = []
    found_gt = initial_vertex_id == ground_truth_id

    candidates = []  # heap of vertices from smallest predicted distance to largest
    heappush(candidates, (initial_distance, initial_vertex_id))
    ef_top = []  # heap of top-ef vertices from largest predicted distance to smallest. Used for pruning
    heappush(ef_top, (-initial_distance, initial_vertex_id))

    visited_ids = {initial_vertex_id}  # a set of vertices already visited by graph walker

    neg_lower_bound_distance, lower_bound_vertex_id = nsmallest(1, ef_top)[0]
    lower_bound_distance = -neg_lower_bound_distance
    dcs, num_hops = 1, 1

    while len(candidates) != 0:
        # record which candidates were available for selection (for loss)
        if not found_gt:
            training_history.append(dict(name='select_next', found_gt=found_gt,
                                         candidate_vertices=[v for _, v in candidates]))

        # 1. pop vertex according to graph walker
        estimated_distance, vertex_id = heappop(candidates)

        if np.isfinite(hnsw.ef):
            training_history.append(dict(
                name='early_termination', found_gt=found_gt,
                chosen_vertex=vertex_id, lower_bound=lower_bound_vertex_id,
            ))
            if estimated_distance > lower_bound_distance: break

        # 2. gather all next vertices
        neighbor_ids = [neighbor_id for neighbor_id in hnsw.graph.edges[vertex_id]
                        if neighbor_id not in visited_ids]

        visited_ids.update(neighbor_ids)

        # 3. compute distances and add all neighbors to candidates
        if len(neighbor_ids) > 0:
            neighbor_vectors = np.stack([vertex_id_to_vectors[ix] for ix in neighbor_ids])
            distances = hnsw.distance_for_routing(query_vector[None], neighbor_vectors)
        else:
            distances = []

        # add penalty for pruning neighbors from heap by lower bound
        # this penalty will only be applied to edges that are closer to gt than best vertex in heap
        penalty_for_pruning = dict(name='do_not_prune_optimal', neighbors=[], lower_bounds=[])
        had_gt_before = found_gt

        for distance, neighbor_id in zip(map(float, distances), neighbor_ids):
            if len(ef_top) > hnsw.ef:
                penalty_for_pruning['neighbors'].append(neighbor_id)
                penalty_for_pruning['lower_bounds'].append(lower_bound_vertex_id)

            if sampling_temperature != 0:
                noise = -np.log(-np.log(np.random.uniform(0.0, 1.0) + eps) + eps)
                distance -= noise * sampling_temperature

            if distance < lower_bound_distance or len(ef_top) < hnsw.ef:
                heappush(candidates, (distance, neighbor_id))
                heappush(ef_top, (-distance, neighbor_id))

                if len(ef_top) > hnsw.ef:
                    _, pruned_vertex_id = heappop(ef_top)

                    # loss: make sure we didn't prune gt vertex
                    if pruned_vertex_id == ground_truth_id:
                        training_history.append(dict(
                            name='keep_gt_in_heap',
                            positive_ids=[pruned_vertex_id],
                            negative_ids=[nsmallest(1, ef_top)[0][1]],
                        ))

                neg_lower_bound_distance, lower_bound_vertex_id = nsmallest(1, ef_top)[0]
                lower_bound_distance = -neg_lower_bound_distance

                # if we found gt, we may want to terminate
                found_gt = found_gt or (neighbor_id == ground_truth_id)
                if found_gt and terminate_on_gt: break

            else:
                pass  # pruned by lower bound

            # early stopping by dcs
            dcs += 1
            if dcs >= hnsw.max_dcs: break

        if not had_gt_before and len(penalty_for_pruning['neighbors']) > 0:
            training_history.append(penalty_for_pruning)

        num_hops += 1
        if num_hops >= hnsw.max_hops: break
        if dcs >= hnsw.max_dcs: break
        if found_gt and terminate_on_gt: break

    # make sure gt is in top-k of the resulting structure
    verification_top = nlargest(hnsw.top_vertices_for_verification, ef_top)
    vertices_for_verification = [chosen_vertex_id for _neg_distance, chosen_vertex_id in verification_top]
    vertices_for_verification = set(vertices_for_verification)
    non_top_vertices = set(v for _, v in ef_top if v not in vertices_for_verification)

    training_history.append(dict(
        name='select_gt', k=hnsw.top_vertices_for_verification, ef_top_heap=ef_top, gt=ground_truth_id,
        top_vertices=list(vertices_for_verification), non_top_vertices=list(non_top_vertices)
    ))
    return dict(dcs=dcs, num_hops=num_hops, training_history=training_history, found_gt=found_gt,
                initial_vertex_id=initial_vertex_id, ground_truth_id=ground_truth_id,
                visited_ids=visited_ids)


def compute_expert_supervision(hnsw, query, record, vertex_id_to_dist,
                               verification_ratio=0.5, select_best_visited=True, **kwargs):
    """
    takes training record as generated by sample_training_record,
    computes a list of training objectives, each defined as a sum of components:
        weight \cdot (\log{ sum_{v_i \in positive_ids} exp(-d(q, v_i)) } -
          - \log{ sum_{v_j \in concat(positive_ids, negative_ids) } exp(-d(q, v_k)) })
    Such objective essentially computes logp(any positive > all negatives) under gumbel noise

    :param vertex_id_to_dist: a dict {vertex id -> distance to gt} for all vertex ids in record['visited_ids']
        also necessarily including v0 and gt
    :param verify_only_if_visited_gt: if True, adds verification loss only if target vertex is visited
    :param verification_ratio: the total weight of all verification objectives equals this
        (whereas the total weight of all other objectives equals 1 - verification_ratio)
    :returns: dict(query, objectives) where objectives is a list of dictionaries,
        each element is a dictionary containing {
        'name': tag for ease of debugging
        'positive_ids': a list of vertex ids to be included in numerator AND denominator
        'negative ids': a list of vertex ids to be included in denominator only
        'weight': multiplicative coefficient
    }
    """

    # compute all objectives
    objectives = []
    for event in record['training_history']:
        if event['name'] == 'select_next':
            # At each step learn to select an optimal vertex from heap
            # an optimal vertex is one of the vertices that have minimal distance to gt in terms of hops
            if event['found_gt']: continue
            candidate_vertices = event['candidate_vertices']
            candidate_distances = [vertex_id_to_dist[v] for v in candidate_vertices]
            optimal_distance = min(candidate_distances)
            objectives.append(dict(
                name=event['name'],
                positive_ids=[v for v, d in zip(candidate_vertices, candidate_distances)
                              if d <= optimal_distance],
                negative_ids=[v for v, d in zip(candidate_vertices, candidate_distances)
                              if d > optimal_distance],
            ))
        elif event['name'] == 'select_gt':
            # make sure ground_truth_id is one of the vertices returned for verification
            # if it isn't found, make
            candidate_vertices = event['top_vertices'] + event['non_top_vertices']
            candidate_distances = [vertex_id_to_dist[v] for v in candidate_vertices]
            if select_best_visited:
                best_visited_disance = min(candidate_distances)
                positive_ids = {v for v, d in zip(candidate_vertices, candidate_distances)
                                if d <= best_visited_disance}
            else:
                positive_ids = {event['gt']}

            negative_ids = {v for v in event['non_top_vertices'] if v not in positive_ids}

            objectives.append(dict(
                name=event['name'],
                positive_ids=list(positive_ids),
                negative_ids=list(negative_ids),
            ))
        else:
            raise NotImplementedError("Objective '{}' isn't implemented, make sure ef == inf"
                                      .format(event['name']))

    assert len(objectives) == len(record['training_history'])

    # compute weights s.t. all verification weights add up to verification_ratio
    # and all non-verification loss adds up to (1.0 - verification_ratio)

    verification_rows = [row for row in objectives if row['name'] == 'select_gt']
    routing_weight = 1. / max(1, len(objectives) - len(verification_rows)) * (1 - verification_ratio)
    verification_weight = 1. / max(1, len(verification_rows)) * verification_ratio
    for row in objectives:
        row['weight'] = verification_weight if ('gt' in row['name']) else routing_weight
    return dict(query=query, objectives=objectives)
