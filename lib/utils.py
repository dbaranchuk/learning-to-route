from sklearn.neighbors import NearestNeighbors
from struct import pack, unpack
from collections import defaultdict
from contextlib import contextmanager
import bokeh.models as bm
import bokeh.plotting as pl
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

"""
                        KNN brute-force search
"""


def knn(base, queries, n_jobs=1, batch_size=1000, n_neighbors=1):
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', n_jobs=n_jobs)
    knn.fit(base)
    idxs = []
    if len(queries) >= batch_size:
        batches_queries = queries.split(len(queries) // batch_size)
    else:
        batches_queries = [queries]

    for batch_queries in batches_queries:
        idx = knn.kneighbors(batch_queries)[1]
        idxs.append(torch.tensor(idx))
    return torch.cat(idxs, 0).view(-1, n_neighbors)


"""
                  IO Utils
"""


def read_fvecs(filename, max_size=None):
    max_size = max_size or float('inf')
    with open(filename, "rb") as f:
        vecs = []
        while True:
            header = f.read(4)
            if not len(header):
                break
            dim, = unpack('<i', header)
            vec = unpack('f' * dim, f.read(4 * dim))
            vecs.append(vec)
            if len(vecs) >= max_size:
                break
    return np.array(vecs, dtype="float32")


def read_ivecs(filename, max_size=None):
    max_size = max_size or float('inf')
    with open(filename, "rb") as f:
        vecs = []
        while True:
            header = f.read(4)
            if not len(header):
                break
            dim, = unpack('<i', header)
            vec = unpack('i' * dim, f.read(4 * dim))
            vecs.append(vec)
            if len(vecs) >= max_size:
                break
    return np.array(vecs, dtype="int32")


def read_edges(filename, max_size=None):
    max_size = max_size or float('inf')
    with open(filename, "rb") as f_edges:
        edges = defaultdict(list)
        while True:
            header = f_edges.read(4)
            if not len(header):
                break
            dim, = unpack('<i', header)
            vec = unpack('i' * dim, f_edges.read(4 * dim))
            edges[len(edges)] = vec
            if len(edges) >= max_size:
                break
    return edges


def write_fvecs(filename, vecs):
    with open(filename, "wb") as f:
        for vec in vecs:
            dim = len(vec)
            f.write(pack('<i', dim))
            f.write(pack('f' * dim, *list(vec)))


def write_ivecs(filename, vecs):
    with open(filename, "wb") as f:
        for vec in vecs:
            dim = len(vec)
            f.write(pack('<i', dim))
            f.write(pack('i' * dim, *list(vec)))


def read_info(filename):
    info = {}
    with open(filename, "rb") as f:
        info['maxelements_'], = unpack('<Q', f.read(8))
        info['enterpoint_node'], = unpack('<i', f.read(4))
        info['data_size'], = unpack('<Q', f.read(8))
        info['offset_data'], = unpack('<Q', f.read(8))
        info['size_data_per_element'], = unpack('<Q', f.read(8))
        info['M_'], = unpack('<Q', f.read(8))
        info['maxM_'], = unpack('<Q', f.read(8))
        info['size_links_level0'], = unpack('<Q', f.read(8))
        info['size_links_per_element'], = unpack('<Q', f.read(8))

        info['elementLevels'] = []
        info['max_level'] = 0
        info['level_edges'] = []
        for i in range(info['maxelements_']):
            linklist_size, = unpack('<i', f.read(4))
            level_edges = {}
            if linklist_size == 0:
                info['elementLevels'].append(0)
            else:
                info['elementLevels'].append(linklist_size // info['size_links_per_element'])
                for level in range(info['elementLevels'][-1]):
                    bytes = f.read(info['size_links_per_element'])
                    size = int(unpack('<H', bytes[:2])[0])
                    level_edges[level] = unpack(size * 'i', bytes[2: size*4+2])

            info['level_edges'].append(level_edges)

        info['max_level'] = max(info['elementLevels'])
    return info


def read_path_cache(filename):
    path_cache = {}
    with open(filename, "rb") as f:
        while True:
            header = f.read(12)
            if not len(header):
                break
            from_id, to_id, size = unpack("III", header)
            data = unpack(size * "I", f.read(4 * size))
            path_cache[(from_id, to_id)] = {data[i]: data[i + 1] for i in range(0, size, 2)}
    return path_cache


"""
                graph visualization
"""


def draw_graph(x, y, edges,
               radius=10, vertex_alpha=0.25, vertex_color='blue',  # vertex params
               edge_width=1, edge_alpha=0.15, edge_color='gray',   # edge params
               width=600, height=400, show=True, **kwargs):
    """ draws an interactive plot for data points with auxilirary info on hover """
    fig = pl.figure(active_scroll='wheel_zoom', width=width, height=height)

    # edges
    edges_ij = [(from_i, to_i) for from_i, to_ix in edges.items() for to_i in to_ix]

    def _select_edges(field):
        if isinstance(field, dict):
            return [field[from_i][to_i] for from_i, to_i in edges_ij]
        else:
            return [field] * len(edges_ij)

    edge_source = bm.ColumnDataSource({'xx': x[edges_ij].tolist(), 'yy': y[edges_ij].tolist(),
                                       'alpha': _select_edges(edge_alpha),
                                       'color': _select_edges(edge_color),
                                       'width': _select_edges(edge_width),
                                       })
    fig.multi_line('xx', 'yy', color='color', line_width='width', alpha='alpha',
                   source=edge_source)

    # vertices
    def _maybe_repeat(x, size):
        if not hasattr(x, '__len__') or len(x) != size:
            x = [x] * size
        return x

    vertex_source = bm.ColumnDataSource({'x': x, 'y': y,
                                         'color': _maybe_repeat(vertex_color, len(x)),
                                         'alpha': _maybe_repeat(vertex_alpha, len(x)),
                                         **kwargs})
    fig.scatter('x', 'y', size=radius, color='color', alpha='alpha', name='vertices',
                source=vertex_source)

    fig.add_tools(bm.HoverTool(tooltips=[(key, "@" + key) for key in kwargs.keys()],
                               names=['vertices']))
    if show:
        pl.show(fig)
    return fig


def iterate_minibatches(*tensors, batch_size=4096, shuffle=True, cycle=True, **kw):
    while True:
        yield from DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)
        if not cycle:
            break


"""
    global variables: set or get interpreter-level variables. 
    Used for efficient multiprocessing
"""
GLOBAL_VARIABLES = None


@contextmanager
def global_variables(**variables):
    global GLOBAL_VARIABLES
    assert GLOBAL_VARIABLES is None, "global variables already set"
    GLOBAL_VARIABLES = variables
    try:
        yield
    finally:
        GLOBAL_VARIABLES = None


def require_variables(*variables):
    assert GLOBAL_VARIABLES is not None, "global variables not set"
    return [GLOBAL_VARIABLES[key] for key in variables]
