import networkx as nx
import utils
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import json

def load_communities():
    with open ("multiplex/data/files/politicsuk.communities", 'r') as f:
        lines = f.readlines()

    communities = {}

    for line in lines:
        elements = line.strip().split(": ")
        community_id = elements[0]
        nodes = list(map(int, elements[1].split(",")))
        for node in nodes:
            communities[node] = community_id
    
    return communities

def load_layers():
    layers = ['follows', 'mentions', 'retweets']
    graphs = []

    for layer in layers:
        with open(f"multiplex/data/files/politicsuk_{layer}.mtx", 'r') as f:
            lines = f.readlines()

        edges = [tuple(map(int, line.strip().split())) for line in lines[1:]]
        
        G = nx.DiGraph()
        for u, v, w in edges:
            G.add_edge(u, v)

        graphs.append(G)
    
    return graphs

def preprocessing():
    graphs = load_layers()

    common_nodes = set.intersection(*[set(graph.nodes()) for graph in graphs])
    graphs = [graph.subgraph(common_nodes).copy() for graph in graphs]

    communities = load_communities()
    other_nodes = [node for node, community in communities.items() if community == 'other']
    common_nodes = common_nodes - set(other_nodes)

    for graph in graphs:
        graph.remove_nodes_from(other_nodes)

    new_labels = range(len(common_nodes))
    mapping = dict(zip(sorted(common_nodes), new_labels))
    for i in range(len(graphs)):
        graphs[i] = nx.relabel_nodes(graphs[i], mapping)
        
    utils.save_pickle(f"multiplex/data/graphs/politicsuk/politicsuk_follows_graph", graphs[0])
    utils.save_pickle(f"multiplex/data/graphs/politicsuk/politicsuk_mentions_graph", graphs[1])
    utils.save_pickle(f"multiplex/data/graphs/politicsuk/politicsuk_retweets_graph", graphs[2])

    communities = {node: community for node, community in communities.items() if node in common_nodes}
    label_to_int = {'conservative':0, 'labour':1, 'libdem':2, 'snp':3}
    community_labels = [label_to_int[communities[node]] for node in sorted(common_nodes)]

    utils.save_pickle("multiplex/data/files/politicsuk_ground_truth", community_labels)

    for i, graph in enumerate(graphs):
        print(f"Graph {i+1} - {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"Is directed: {nx.is_directed(graph)}")
        print(f"Is connected: {nx.is_connected(graph.to_undirected())}")
        print(f"Average degree: {sum(dict(graph.degree()).values()) / graph.number_of_nodes()}\n")

def compare():
    gt_labels = utils.load_pickle("multiplex/data/files/politicsuk_ground_truth")
    path = "multiplex/output/politicsuk_weighted_10000_4/clusters.json"
    predicted_labels = utils.get_predicted_labels(json.load(open(path, 'r')))

    ari = adjusted_rand_score(gt_labels, predicted_labels)
    nmi = normalized_mutual_info_score(gt_labels, predicted_labels)
    print(f"Adjusted Rand Index: {ari}")
    print(f"Normalized Mutual Information: {nmi}")
