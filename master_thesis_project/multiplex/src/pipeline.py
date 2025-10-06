import networkx as nx
import numpy as np
import random
from collections import deque
from sklearn.cluster import KMeans
from collections import Counter

from src import utils
from src import create_weights

def run(dataset, type, n_simulations, threshold):
    layers = utils.load_layers(dataset)
    
    if type == 'random':
        w_layers = create_weights.w_random_multiple(layers)
    elif type == 'uniform':
        w_layers = create_weights.w_uniform_multiple(layers)
    elif type == 'weighted':
        w_layers = create_weights.w_weighted_multiple(layers)
    elif type == 'trivalency':
        w_layers = create_weights.w_trivalency_multiple(layers)

    return live_edge_simulation(n_simulations, w_layers, threshold)

def live_edge_simulation(n_simulations, layers, threshold):
    n_nodes = len(layers[0].nodes())
    IA = np.zeros((n_nodes, n_nodes))

    for i in range(n_simulations):
        print(f'Simulation {i + 1}/{n_simulations} ...')
        live_edge_graphs = create_live_edge_graphs(layers)

        for seed in range(n_nodes):
            activated_nodes = get_activated_nodes(live_edge_graphs, seed, threshold)
            for node in activated_nodes:
                IA[node][seed] += 1

    IA /= n_simulations
    return IA

def create_live_edge_graphs(layers):
    graphs = []

    for layer in layers:
        live_edge_graph = nx.DiGraph()

        for node in layer.nodes:
            incoming_edges = list(layer.in_edges(node, data=True))
            if len(incoming_edges) == 0 or incoming_edges is None:
                live_edge_graph.add_node(node)
                continue

            total_weight = sum(data['weight'] for _, _, data in incoming_edges)
            probabilities = [data['weight'] for _, _, data in incoming_edges]
            probabilities.append(1 - total_weight)
            chosen_index = random.choices(range(len(probabilities)), weights=probabilities, k=1)[0]

            if chosen_index == len(probabilities) - 1:
                live_edge_graph.add_node(node)
                continue

            start_node, end_node, _ = incoming_edges[chosen_index]
            live_edge_graph.add_edge(start_node, end_node)

        graphs.append(live_edge_graph)

    return graphs

def get_activated_nodes(live_edge_graphs, seed, threshold):
    reachables = []

    for graph in live_edge_graphs:
        reachable = set()
        queue = deque([seed])

        while queue:
            node = queue.popleft()
            if node not in reachable:
                reachable.add(node)
                for neighbor in graph.successors(node):
                    if neighbor not in reachable:
                        queue.append(neighbor)

        reachables.append(list(reachable))

    id_counter = Counter()
    for lst in reachables:
        unique_nodes = set(lst)
        id_counter.update(unique_nodes)

    return [id_ for id_, count in id_counter.items() if count / len(reachables) >= threshold]
    
# Clustering
def clustering(n_clusters, IA):
    # normalization
    IA = normalize_offdiag(IA)

    labels = KMeans(n_clusters=n_clusters, random_state=1, n_init='auto').fit_predict(IA)

    clusters = {}
    for node in range(len(labels)):
        label = int(labels[node])
        if label in clusters:
            value = clusters[label]
            value.append(node)
            clusters[label] = value
        else:
            clusters[label] = [node]

    clusters = {key: clusters[key] for key in sorted(clusters)}
    return clusters

def normalize_offdiag(A):
    n = A.shape[0]
    B = A.copy().astype(float)
    np.fill_diagonal(B, 0.0)
    row_sum = B.sum(axis=1)
    mu = row_sum / (n - 1)

    sq = ((B - mu[:,None])**2).sum(axis=1)
    sigma = np.sqrt(sq / (n - 1))
    sigma[sigma == 0] = 1.0
    B = (B - mu[:,None]) / sigma[:,None]
    np.fill_diagonal(B, 0.0)

    return B