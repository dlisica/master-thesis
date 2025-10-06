import networkx as nx
import utils
from tabulate import tabulate

def calculate_density(G):
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    max_edges = n_nodes * (n_nodes - 1)
    return n_edges / max_edges if max_edges > 0 else 0

def create_layers():
    layer0_edges = []
    with open('multiplex/data/files/flickr_friendship.txt', 'r') as file:
        for line in file:
            u, v = line.strip().split()
            layer0_edges.append((int(u), int(v)))

    layer1_edges = []
    with open('multiplex/data/files/flickr_tag_similarity.txt', 'r') as file:
        for line in file:
            u, v = line.strip().split()
            layer1_edges.append((int(u), int(v)))

    layer0_graph = nx.Graph()
    layer0_graph.add_edges_from(layer0_edges)
    layer0_graph = utils.to_directed(layer0_graph)

    layer1_graph = nx.Graph()
    layer1_graph.add_edges_from(layer1_edges)
    layer1_graph = utils.to_directed(layer1_graph)

    print(f"Friendship: {layer0_graph.number_of_nodes()} nodes, {layer0_graph.number_of_edges()} edges")
    print(f"Tag-similarity: {layer1_graph.number_of_nodes()} nodes, {layer1_graph.number_of_edges()} edges")

    print(f"Friendship density: {calculate_density(layer0_graph):.4f}")
    print(f"Tag-similarity density: {calculate_density(layer1_graph):.4f}")

    utils.save_pickle('multiplex/data/graphs/flickr/flickr_friendship_graph', layer0_graph)
    utils.save_pickle('multiplex/data/graphs/flickr/flickr_tag_similarity_graph', layer1_graph)

def cluster_statistics():
    ground_truth = utils.load_ground_truth('flickr')
    total_nodes = len(ground_truth)

    cluster_dict = {}
    for node_id, cluster_id in enumerate(ground_truth):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(node_id)

    cluster_data = []
    for cluster_id, nodes in sorted(cluster_dict.items()):
        count = len(nodes)
        percentage = (count / total_nodes) * 100
        cluster_data.append([cluster_id, count, f"{percentage:.2f}%"])

    print(tabulate(cluster_data, headers=["Cluster Number", "Node Count", "Percentage"], tablefmt="grid"))

create_layers()