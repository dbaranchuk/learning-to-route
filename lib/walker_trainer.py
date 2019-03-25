import joblib
import time
import torch
import numpy as np
from heapq import heappush, heappop, nsmallest
from tensorboardX import SummaryWriter
from datetime import datetime

import lib
from lib.utils import global_variables, require_variables, iterate_minibatches
from lib.nn_utils import get_device_of


class SupervisedWalkerTrainer:
    def __init__(self, agent, hnsw, path_cache, writer=None,
                 Optimizer=lambda params: torch.optim.Adam(params, lr=1e-4, amsgrad=True),
                 device=None, **learning_rate_opts):
        """
        A class that handles agent training
        :type agent: lib.walker_agent.BaseWalkerAgent
        :type hnsw: lib.hnsw.WalkerHNSW
        :type path_cache: lib.paths.PathCache
        :param device: device to run computations on (string)
        """
        self.agent, self.hnsw, self.path_cache = agent, hnsw, path_cache
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
            for row in rec['logp_history']:
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
            for row in rec['logp_history']:
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
            records = generate_records(
                self.hnsw, self.agent, self.path_cache,
                chunk_queries, chunk_gt, initial_vertices=chunk_initial_vertices,
                n_jobs=n_jobs, device=self.device, **kwargs
            )
            if verbose:
                print("Done!", flush=True)

            for rec in records:
                if len(batch_records) > 0 and total_size + len(rec['logp_history']) > batch_size_max:
                    yield batch_records
                    batch_records, total_size = [], 0

                batch_records.append(rec)
                total_size += len(rec['logp_history'])


def generate_records(hnsw, agent, path_cache, queries, ground_truth,
                     initial_vertices=None, batch_size=None, n_jobs=-1, **kwargs):
    """ Generate a batch of training records by pre-computing graph vertices and sampling trajectories """
    query_vectors, vertex_id_to_vectors = prepare_vectors(agent, hnsw.graph, queries,
                                                          batch_size=batch_size, **kwargs)
    if initial_vertices is None:
        initial_vertices = [hnsw.get_initial_vertex_id(q, **kwargs) for q in queries]

    with global_variables(hnsw=hnsw, path_cache=path_cache, vertex_id_to_vectors=vertex_id_to_vectors):
        make_sample_job = joblib.delayed(_sample_job)
        jobs = (make_sample_job(*args, **kwargs)
                for args in zip(queries.data.numpy(), query_vectors, initial_vertices, ground_truth.data.numpy()))
        return joblib.Parallel(n_jobs=n_jobs)(jobs)


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


def _sample_job(query, query_vec, initial_vertex, target_vertex, **kwargs):
    """ A wrapper that samples one trajectory, used with joblib """
    hnsw, path_cache, vertex_id_to_vectors = require_variables('hnsw', 'path_cache', 'vertex_id_to_vectors')
    vertex_id_to_dist = path_cache[initial_vertex, target_vertex]
    record = sample_training_record(
        hnsw, path_cache, query_vec, vertex_id_to_vectors,
        initial_vertex_id=int(initial_vertex),
        vertex_id_to_dist=vertex_id_to_dist,
        **kwargs)
    return dict(record, query=query, gt=target_vertex)


def sample_training_record(hnsw, path_cache, query_vector, vertex_id_to_vectors,
                           *, vertex_id_to_dist, initial_vertex_id,
                           terminate_on_gt=False, verify_only_if_visited_gt=False,
                           sampling_temperature=1.0, eps=1e-6, **kwargs):
    """ Samples EF-search trajectory using pre-computed vectors
    :type hnsw: lib.hnsw.WalkerHNSW
    :type path_cache: lib.paths.PathCache
    :param query_vector: query representation from agent.get_query_vector
    :param vertex_id_to_vectors: dict {vertex -> vector}
    :param vertex_id_to_dist: dict {vertex id -> num hops to target vertex}
    :param initial_vertex_id: vertex id to start from
    :param terminate_on_gt: if True, ends search immediately after visiting target vertex
    :param verify_only_if_visited_gt: if True, adds verification loss only if target vertex is visited
    :param sampling_temperature: coefficient for gumbel noise added to distances.
        temperature=0 -> move exactly as during test, temperature > 0 -> more random moves
        higher temperature encourages exploration
    """
    assert isinstance(initial_vertex_id, (type(None), int))
    assert initial_vertex_id in vertex_id_to_dist

    initial_distance = hnsw.distance_for_routing(query_vector, vertex_id_to_vectors[initial_vertex_id])
    if sampling_temperature != 0:
        noise = -np.log(-np.log(np.random.uniform(0.0, 1.0) + eps) + eps)
        initial_distance -= noise * sampling_temperature

    # Below: a list of training examples:
    # {'mode': LOGP_MODES, 'positive_ids': [...], 'negative_edges': [...]}
    # such that at least one of positive_ids must be closer than all negative_ids
    training_history = []

    candidates = []  # heap of vertices from smallest predicted distance to largest
    heappush(candidates, (initial_distance, initial_vertex_id))
    ef_top = []  # heap of top-ef vertices from largest predicted distance to smallest. Used for pruning
    heappush(ef_top, (-initial_distance, initial_vertex_id))

    visited_ids = {initial_vertex_id}  # a set of vertices already visited by graph walker

    # best visited vertex_id by num hops to gt
    best_visited_actual_distance = vertex_id_to_dist[initial_vertex_id]
    best_visited_vertex_id = initial_vertex_id

    neg_lower_bound_distance, lower_bound_vertex_id = nsmallest(1, ef_top)[0]
    lower_bound_distance = -neg_lower_bound_distance
    dcs, num_hops = 1, 1

    while len(candidates) != 0:

        # 1. pop vertex according to graph walker
        estimated_distance, vertex_id = heappop(candidates)

        # if budget setup walk until the end
        if np.isfinite(hnsw.ef) and estimated_distance > lower_bound_distance: break

        # 2. gather all next vertices
        neighbor_ids = [neighbor_id for neighbor_id in hnsw.graph.edges[vertex_id]
                        if neighbor_id not in visited_ids and neighbor_id in vertex_id_to_dist]

        # update visited ids
        visited_ids.update(neighbor_ids)

        # 3. compute distances and add all neighbors to candidates
        if len(neighbor_ids) > 0:
            neighbor_vectors = np.stack([vertex_id_to_vectors[ix] for ix in neighbor_ids])
            distances = hnsw.distance_for_routing(query_vector[None], neighbor_vectors)
        else:
            distances = []

        optimal_neighbors_with_lower_bounds = []  # [(neighbor_id, lower_bound_id)]
        best_visited_actual_distance_before_expand = best_visited_actual_distance

        for distance, neighbor_id in zip(map(float, distances), neighbor_ids):
            if vertex_id_to_dist[neighbor_id] < best_visited_actual_distance_before_expand:
                if len(ef_top) > hnsw.ef:
                    optimal_neighbors_with_lower_bounds.append((neighbor_id, lower_bound_vertex_id))

            if sampling_temperature != 0:
                noise = -np.log(-np.log(np.random.uniform(0.0, 1.0) + eps) + eps)
                distance -= noise * sampling_temperature

            if distance < lower_bound_distance or len(ef_top) < hnsw.ef:
                heappush(candidates, (distance, neighbor_id))
                heappush(ef_top, (-distance, neighbor_id))

                if len(ef_top) > hnsw.ef:
                    _, pruned_vertex_id = heappop(ef_top)

                    # loss: make sure we didn't prune gt vertex
                    if vertex_id_to_dist[pruned_vertex_id] == 0:
                        training_history.append(dict(
                            name='keep_gt_in_heap', level=0,
                            positive_ids=[pruned_vertex_id],
                            negative_ids=[nsmallest(1, ef_top)[0][1]],
                        ))

                neg_lower_bound_distance, lower_bound_vertex_id = nsmallest(1, ef_top)[0]
                lower_bound_distance = -neg_lower_bound_distance
            else:
                pass  # pruned by lower bound

            # maintain the nearest visited vertex
            if vertex_id_to_dist[neighbor_id] < best_visited_actual_distance:
                best_visited_actual_distance = vertex_id_to_dist[neighbor_id]
                best_visited_vertex_id = neighbor_id

            # early stopping by dcs
            dcs += 1
            if dcs >= hnsw.max_dcs: break

        # 4. gather all optimal and all suboptimal choices for the next steps

        # 4.1 If we haven't found gt (reference answer) yet,
        # optimal action is to expand the vertex which is closest to answer (or any of several such vertices)
        if best_visited_actual_distance != 0 and len(candidates) > 0:
            positive_ids = {
                vid for _, vid in candidates
                if vertex_id_to_dist[vid] <= best_visited_actual_distance
            }
            negative_ids = {vid for _, vid in candidates
                            if vertex_id_to_dist[vid] > best_visited_actual_distance}

            # Also make sure at least one optimal vertex will not be pruned by lower bound during next [step 1.]
            negative_ids.add(lower_bound_vertex_id)
            if len(positive_ids) > 0 and len(negative_ids) > 0:
                training_history.append(dict(
                    name='select_next', level=vertex_id_to_dist[next(iter(positive_ids))],
                    positive_ids=list(positive_ids),
                    negative_ids=list(negative_ids),
                ))
            else:
                pass  # at this point fringe does not contain any vertices better than what we've visited

        # 4.2 Also if new visited_ids from contained something better than our current candidates,
        # make sure we didn't just prune it. At least one optimal candidate must exceed its lower bound
        # note: this is only possible if best_visited_actual_distance != 0
        if len(optimal_neighbors_with_lower_bounds):
            positive_ids, negative_ids = zip(*optimal_neighbors_with_lower_bounds)
            training_history.append(dict(
                name='do_not_prune_optimal', level=vertex_id_to_dist[positive_ids[0]],
                positive_ids=list(set(positive_ids)),
                negative_ids=list(set(negative_ids)),
            ))
            # Note: we use logp-any as a proxy of the actual loss, which should be pairwise sigmoid log-p

        if terminate_on_gt and best_visited_actual_distance == 0:
            # we have already found gt, nothing else required
            # TODO: learn to terminate search ASAP?
            break

        num_hops += 1
        if num_hops >= hnsw.max_hops: break
        if dcs >= hnsw.max_dcs: break

    # make sure hnsw finds best vertex
    if not verify_only_if_visited_gt or best_visited_actual_distance == 0:
        # found gt IF best vertex is in ef_top and is one of the vertices used for selection
        positive_ids = {vid for _, vid in ef_top
                        if vertex_id_to_dist[vid] == best_visited_actual_distance}

        num_bottom_vertices = len(ef_top) - hnsw.top_vertices_for_verification
        below_top = nsmallest(num_bottom_vertices, ef_top)
        negative_ids = {vid for _, vid in below_top if vid not in positive_ids}

        training_history.append(dict(
            name='select_gt', level=best_visited_actual_distance,
            positive_ids=list(positive_ids),
            negative_ids=list(negative_ids),
        ))

    return compute_record_weights(dict(dcs=dcs, num_hops=num_hops, logp_history=training_history,
                                       best_visited_distance=best_visited_actual_distance,
                                       best_visited_vertex_id=best_visited_vertex_id))


def compute_record_weights(record, verification_ratio=0.5):
    """
    Computes weights s.t. all verification weights add up to verification_ratio
    and all non-verification loss adds up to (1.0 - verification_ratio)
    """
    verification_rows = [row for row in record['logp_history'] if row['name'] == 'select_gt']
    routing_weight = 1. / max(1, len(record['logp_history']) - len(verification_rows)) * (1 - verification_ratio)
    verification_weight = 1. / max(1, len(verification_rows)) * verification_ratio
    for row in record['logp_history']:
        row['weight'] = verification_weight if ('gt' in row['name']) else routing_weight
    return record
