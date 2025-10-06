import pickle
import networkx as nx
import random

def load_pickle(path):
    with open(path, 'rb') as pickle_file:
        G = pickle.load(pickle_file)
    return G

def save_pickle(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file)

def load_layers(dataset):
    if dataset == 'flickr':
        l0 = load_pickle('multiplex/data/graphs/flickr/flickr_friendship_graph')
        l1 = load_pickle('multiplex/data/graphs/flickr/flickr_tag_similarity_graph')
        return [l0, l1]
    elif dataset == 'politicsuk':
        l0 = load_pickle('multiplex/data/graphs/politicsuk/politicsuk_follows_graph')
        l1 = load_pickle('multiplex/data/graphs/politicsuk/politicsuk_mentions_graph')
        l2 = load_pickle('multiplex/data/graphs/politicsuk/politicsuk_retweets_graph')
        return [l0, l1, l2]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def to_directed(G):
    if nx.is_directed(G):
        print(f'Graph is already directed')
        print()
        return G

    directed = nx.DiGraph()
    directed.add_nodes_from(G.nodes(data=True))

    for u, v in G.edges:
        if random.choice([True, False]):
            directed.add_edge(u, v)
        else:
            directed.add_edge(v, u)
    
    return directed

def create_layer(matrix):
    G = nx.Graph()
    rows, cols = matrix.shape
    
    G.add_nodes_from(range(rows))
    
    for r in range(rows):
        for c in range(cols):
            if matrix[r, c] != 0 and r != c:
                G.add_edge(r, c)

    return G

def load_ground_truth(dataset):
    if dataset == 'flickr':
        file_path = 'multiplex/data/files/flickr_ground_truth.txt'
    elif dataset == 'politicsuk':
        return load_pickle("multiplex/data/files/politicsuk_ground_truth")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    with open(file_path, 'r') as file:
        lines = file.readlines()
        ground_truth = [int(line.strip()) for line in lines]

    return ground_truth

def get_predicted_labels(clusters):
    n_nodes = sum(len(cluster) for cluster in clusters.values())
    predicted_labels = [None] * n_nodes

    for label,cluster in clusters.items():
        for node in cluster:
            predicted_labels[int(node)] = label

    if None in predicted_labels:
        raise ValueError("Some nodes do not have a predicted label assigned.")

    return predicted_labels
