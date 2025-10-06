import networkx as nx
import numpy as np

from sklearn.cluster import SpectralClustering, KMeans
from scipy.sparse import csr_matrix
from networkx.algorithms.community import asyn_fluidc
from community import community_louvain

import utils

def spectral_clustering(dataset, k=2):
    G = utils.load_graph(dataset)
    n = G.number_of_nodes()
    nodes = list(range(n))

    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight=None, dtype=int, format="csr")
    S = A + A.T

    S = S.tolil()
    S.setdiag(0)
    S = S.tocsr()

    S = (S > 0).astype(float)
    S.eliminate_zeros()
    S = csr_matrix(S)

    sc = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=1,
    )
    labels = sc.fit_predict(S)
    return labels


def fc_clustering(dataset, k=2):
    G = utils.load_graph(dataset)
    n = G.number_of_nodes()

    U = nx.Graph()
    U.add_nodes_from(range(n))
    U.add_edges_from(G.edges())

    if U.number_of_nodes() > 0 and not nx.is_connected(U):
        comps = [list(c) for c in nx.connected_components(U)]
        for i in range(len(comps) - 1):
            U.add_edge(comps[i][0], comps[i + 1][0])

    k = max(1, min(k, n))
    comms = list(asyn_fluidc(U, k=k, seed=1, max_iter=100))

    labels = np.empty(n, dtype=int)
    for cid, members in enumerate(comms):
        for u in members:
            labels[u] = cid

    return labels


def louvain_clustering(dataset):
    G = utils.load_graph(dataset)
    n = G.number_of_nodes()

    U = nx.Graph()
    U.add_nodes_from(range(n))
    U.add_edges_from(G.edges())

    partition = community_louvain.best_partition(
        U, weight="weight", resolution=1.0, random_state=1
    )

    labels = np.empty(n, dtype=int)
    for i in range(n):
        labels[i] = partition.get(i, 0)

    return labels

def cp_clustering(dataset):
    G = utils.load_graph(dataset)
    n = G.number_of_nodes()

    U = nx.Graph()
    U.add_nodes_from(range(n))
    U.add_edges_from(G.edges())

    if U.number_of_nodes() == 0:
        return np.array([], dtype=int)

    if U.number_of_edges() == 0:
        return np.zeros(n, dtype=int)

    core_num = nx.core_number(U)
    max_core = max(core_num.values()) if core_num else 0

    labels = np.zeros(n, dtype=int)
    for i in range(n):
        if core_num.get(i, 0) == max_core:
            labels[i] = 1

    return labels
