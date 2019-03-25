import warnings

from .utils import knn, read_edges, read_fvecs, read_info, read_ivecs
import numpy as np
import torch


class Graph:
    def __init__(self, vertices_path, edges_path,
                 train_queries_path, test_queries_path,
                 train_gt_path=None, test_gt_path=None,
                 vertices_size=None, train_queries_size=None, test_queries_size=None,
                 train_queries_sample_size=None,
                 info_path=None,
                 ground_truth_n_neighbors=1, normalization='global',
                 initial_vertex_id=0, seed=42):
        """
        Graph is a data class that stores all CONSTANT data about the graph: vertices, edges, etc.
        :param train_queries_sample_size: samples this many queries from train set at random
        :param ground_truth_n_neighbors: finds this many nearest neighbors for ground truth ids
        :param initial_vertex_id: starts search from this vertex
        """
        self.initial_vertex_id = initial_vertex_id
        vertices = torch.tensor(read_fvecs(vertices_path, vertices_size))
        self.edges = read_edges(edges_path, vertices.shape[0])

        if info_path is not None:
            info = read_info(info_path)
            self.level_edges = info['level_edges']
            self.initial_vertex_id = info['enterpoint_node']
            self.max_level = info['max_level']
        else:
            self.initial_vertex_id = initial_vertex_id
            self.max_level = 0

        train_queries = read_fvecs(train_queries_path, train_queries_size)
        if train_queries_sample_size is not None:
            np.random.seed(seed)
            train_queries = train_queries[np.random.choice(train_queries.shape[0], replace=False,
                                                           size=train_queries_sample_size
                                                          )]
        train_queries = torch.tensor(train_queries)
        test_queries = torch.tensor(read_fvecs(test_queries_path, test_queries_size))

        if normalization == 'none':
            warnings.warn("Data not normalized, individual norms:",
                          (vertices ** 2).sum(-1).cpu().data.numpy())
            normalize = lambda v: v
        elif normalization == 'global':
            mean_norm = ((vertices ** 2).sum(-1) ** 0.5).mean().item()
            normalize = lambda v: v / mean_norm
        elif normalization == 'instance':
            normalize = lambda v: v / (v ** 2).sum(-1, keepdim=True)
        else:
            raise ValueError("normalization parameter must be in ['none', 'global', 'instance']")

        self.vertices, self.train_queries, self.test_queries = \
            map(normalize, [vertices, train_queries, test_queries])

        if train_gt_path is None:
            self.train_gt = knn(self.vertices, self.train_queries, n_neighbors=ground_truth_n_neighbors)
        else:
            self.train_gt = torch.tensor(read_ivecs(train_gt_path, train_queries_size), dtype=torch.long)

        if test_gt_path is None:
            self.test_gt = knn(self.vertices, self.test_queries, n_neighbors=ground_truth_n_neighbors)
        else:
            self.test_gt = torch.tensor(read_ivecs(test_gt_path, test_queries_size), dtype=torch.long)
