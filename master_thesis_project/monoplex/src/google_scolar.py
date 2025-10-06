import networkx as nx
import pickle

import utils

def gs_get_academic_title(number):
    if number == '3':
        return 'professor'
    elif number == '2':
        return 'postdoc'
    elif number == '1':
        return 'student'
    else:
        return 'unknown'

def create_gs_dataset():
    G = nx.DiGraph()
    edges = []
    node_data = {}

    print('Reading edges...')
    with open('monoplex/data/files/gs_edges.txt', 'r') as f:
        for line in f:
            node1, node2 = line.strip().split(',')
            edges.append((int(node1), int(node2)))

    print('Reading info...')
    with open('monoplex/data/files/gs_info.txt', 'r') as f:
        for line in f:
            node_info = line.strip().split()
            node_data[int(node_info[0])] = {
                'total_number_of_citations': int(node_info[1]),
                'h-index': int(node_info[2]),
                'g-index': int(node_info[3]),
                'academic_title': gs_get_academic_title(node_info[4]),
                'computer_science_author': node_info[5] == '1',
                'biology_author': node_info[6] == '1',
                'sociology_author': node_info[7] == '1'
            }

    print('Adding nodes ...')
    for node_id, attributes in node_data.items():
        G.add_node(int(node_id), **attributes)

    print('Adding edges ...')
    G.add_edges_from(edges)

    print('Saving graph ...')
    with open("monoplex/data/graphs/entire_gs_graph", 'wb') as pickle_file:
        pickle.dump(G, pickle_file)

def filter_gs_dataset():
    G = utils.load_pickle(path=f'monoplex/data/graphs/entire_gs_graph')

    for node, attributes in list(G.nodes(data=True)):
        if not attributes.get('computer_science_author', True) or attributes.get('academic_title') == 'unknown':
            G.remove_node(node)

    largest_cc = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc)

    new_labels = {old_label: i for i, old_label in enumerate(subgraph.nodes())}
    subgraph = nx.relabel_nodes(subgraph, new_labels)

    utils.save_pickle('monoplex/data/graphs/gs_graph', subgraph)