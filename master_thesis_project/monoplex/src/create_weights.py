import random
from collections import deque

def w_weighted(graph):
    in_degrees = dict(graph.in_degree())

    for u, v in graph.edges():
        graph[u][v]['weight'] = 1 / in_degrees[v]

    if not validation(graph):
        raise ValueError('Weighted failed!')

    return graph

def w_uniform(graph):
    in_degrees = dict(graph.in_degree())
    max_in_degree_node = max(in_degrees, key=in_degrees.get)
    max_in_degree = in_degrees[max_in_degree_node]

    for u, v in graph.edges():
        graph[u][v]['weight'] = 1 / max_in_degree

    if not validation(graph):
        raise ValueError('Uniform failed!')

    return graph

def w_random(graph):
    for u, v in graph.edges():
        graph[u][v]['weight'] = random.random()

    for node in graph.nodes:
        weights = {}
        for predecessor in graph.predecessors(node):
            edge_data = graph.get_edge_data(predecessor, node)
            weights[predecessor] = edge_data['weight'] if edge_data is not None else None
        incoming_sum = sum(weights.values())
        if incoming_sum <= 1:
            continue
        
        weights = deque(sorted(weights.items(), key=lambda item: item[1], reverse=True))

        while sum(element[1] for element in weights) > 1:
            first = weights.popleft()
            weights.append((first[0], first[1]/2))

        for tuple in weights:
            predecessor = tuple[0]
            weight = tuple[1]
            graph[predecessor][node]['weight'] = weight

    if not validation(graph):
        raise ValueError('Random failed!')
            
    return graph

def w_trivalency(graph):
    values = [0.1, 0.01, 0.001]
    in_degrees = dict(graph.in_degree())
    max_in_degree_node = max(in_degrees, key=in_degrees.get)
    max_in_degree = in_degrees[max_in_degree_node]

    if max_in_degree * values[2] > 1:
        raise ValueError('Trivalency not possible!')

    for u, v in graph.edges():
        graph[u][v]['weight'] = random.choice(values)

    for node in graph.nodes:
        weights = {}
        for predecessor in graph.predecessors(node):
            edge_data = graph.get_edge_data(predecessor, node)
            weights[predecessor] = edge_data['weight'] if edge_data is not None else None
        incoming_sum = sum(weights.values())
        if incoming_sum <= 1:
            continue
        
        weights = deque(sorted(weights.items(), key=lambda item: item[1], reverse=True))

        while sum(element[1] for element in weights) > 1:
            first = weights.popleft()
            if first == values[2]:
                weights.append(first)
            weights.append((first[0], first[1]/10))

        for tuple in weights:
            predecessor = tuple[0]
            weight = tuple[1]
            graph[predecessor][node]['weight'] = weight
    
    if not validation(graph):
        raise ValueError('Trivalency failed!')

    return graph

def validation(graph):
    for node in graph.nodes:
        in_weight = 0
        for predecessor in graph.predecessors(node):
            edge_data = graph.get_edge_data(predecessor, node)
            in_weight += edge_data['weight']
        if in_weight > 1 + 1e-9:
            print(in_weight)
            return False
    
    return True