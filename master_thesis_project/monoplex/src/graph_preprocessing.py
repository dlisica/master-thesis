import utils
import networkx as nx
import random

def to_directed(dataset):
    G = utils.load_graph(dataset)

    print(f'Converting {dataset} graph to directed graph')

    if nx.is_directed(G):
        print(f'Graph {dataset} is already directed')
        print()
        return
    
    #check number of nodes and edges before
    print(f'Number of nodes before: {G.number_of_nodes()}')
    print(f'Number of edges before: {G.number_of_edges()}')

    directed = nx.DiGraph()
    directed.add_nodes_from(G.nodes(data=True))

    for u, v in G.edges:
        if random.choice([True, False]):
            directed.add_edge(u, v)
        else:
            directed.add_edge(v, u)

    #check number of nodes and edges after
    print(f'Number of nodes after: {directed.number_of_nodes()}')
    print(f'Number of edges after: {directed.number_of_edges()}')
    print()
    
    #utils.save_pickle(path=f'monoplex/data/graphs/{dataset}_graph', data=directed)

def main():
    to_directed('cosponsorship')
    to_directed('twitch')
    to_directed('gs')

if __name__ == '__main__':
    main()
