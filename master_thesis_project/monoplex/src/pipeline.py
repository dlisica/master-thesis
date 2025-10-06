import networkx as nx
import numpy as np
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import random
import multiprocessing as mp
from collections import deque
from sklearn.cluster import KMeans
import time

from src import utils
from src import create_weights

NUMBER_OF_PROCESSES = 8

def run(dataset, type, n_simulations):
    graph = utils.load_graph(dataset)

    if type == "ndlib":
        IA, time_per_simulation = ndlib_simulation(n_simulations, graph, NUMBER_OF_PROCESSES)
        return IA, time_per_simulation, None
    
    if type == 'random':
        graph = create_weights.w_random(graph)
    elif type == 'uniform':
        graph = create_weights.w_uniform(graph)
    elif type == 'weighted':
        graph = create_weights.w_weighted(graph)
    elif type == 'trivalency':
        graph = create_weights.w_trivalency(graph) 

    average_weight = np.mean([data['weight'] for _, _, data in graph.edges(data=True)])
    print(f"Average edge weight: {average_weight}")

    IA, time_per_simulation = live_edge_simulation(n_simulations, graph)
    return IA, time_per_simulation, average_weight

# NDLib simulator
def ndlib_simulation(n_simulations, graph, n_processes):
    n_nodes = len(graph.nodes())
    IA = np.zeros((n_nodes, n_nodes))

    simulations_per_process = n_simulations // n_processes
    processes = []
    queue = mp.Queue()

    start_time = time.time()

    print("Creating processes ...")
    for i in range(n_processes):
        p = mp.Process(target=ndlib_simulation_process, args=(graph, n_nodes, simulations_per_process, queue))
        processes.append(p)
        p.start()

    print("Waiting for results ...")
    for index in range(n_processes):
        element = queue.get()
        if element is not None:
            IA += element
            print(f'Process {index}: success')
        else:
            print(f'Process {index}: error')

    print("Waiting for join ...")
    for p in processes:
        p.join()

    time_per_simulation = (time.time() - start_time) / n_simulations

    IA /= n_simulations
    return IA, time_per_simulation

def ndlib_simulation_process(graph, n_nodes, n_simulations, queue):
    IA = np.zeros((n_nodes, n_nodes))
    model = ep.ThresholdModel(graph)

    print('Process start!')

    for _ in range(n_simulations):
        config = mc.Configuration()

        for node in graph.nodes():
            threshold = random.uniform(0, 1)
            config.add_node_configuration("threshold", node, threshold)

        for seed in range(n_nodes):
            model.status = {node: 0 for node in graph.nodes()}
            infected_nodes = [seed]
            config.add_model_initial_configuration("Infected", infected_nodes)
            model.set_initial_status(config)
            
            iteration = model.iteration()
            while iteration['status_delta'][0] != 0 or iteration['status_delta'][1] != 0:
                iteration = model.iteration()
            
            activated = [key for key, value in model.status.items() if value == 1]
            for node in activated:
                IA[node][seed] += 1

    queue.put(IA)
    print('Process done!')

# Live-edge simulator
def live_edge_simulation(n_simulations, graph):
    n_nodes = len(graph.nodes())
    IA = np.zeros((n_nodes, n_nodes))

    start_time = time.time()

    for index in range(n_simulations):
        print(f"Simulation {index + 1}/{n_simulations}")

        live_edge_graph = create_live_edge_graph(graph)

        for seed in range(n_nodes):
            activated_nodes = get_activated_nodes(live_edge_graph, seed)
            for node in activated_nodes:
                IA[node][seed] += 1

    time_per_simulation = (time.time() - start_time) / n_simulations
    IA /= n_simulations

    return IA, time_per_simulation

def create_live_edge_graph(G):
    live_edge_graph = nx.DiGraph()

    for node in G.nodes:
        incoming_edges = list(G.in_edges(node, data=True))
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

    return live_edge_graph

def get_activated_nodes(live_edge_graph, seed):
    reachable = set()
    queue = deque([seed])

    while queue:
        node = queue.popleft()
        if node not in reachable:
            reachable.add(node)
            for neighbor in live_edge_graph.successors(node):
                if neighbor not in reachable:
                    queue.append(neighbor)
    
    return list(reachable)

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
