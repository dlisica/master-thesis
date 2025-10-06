import numpy as np
from src import utils
from src import ploting

def create_report(output_folder, clusters, dataset, time_per_simulation, average_weight):
    graph = utils.load_graph(dataset)

    with open(f"{output_folder}/report.md", 'w') as file:
        file.write('# Information Access Clustering Statistics\n\n')
        file.write(f'Dataset: {dataset}\n')
        file.write(f'Number of clusters: {len(clusters)}\n')
        file.write(f'Time per simulation: {time_per_simulation} seconds\n')
        file.write(f'Average edge weight: {average_weight}\n\n')
        file.write(get_cluster_statistics(dataset, clusters, graph, output_folder))

def get_cluster_statistics(dataset, clusters, graph, output_folder):
    if dataset == 'twitch':
        result = '| Cluster Number | Nodes | Partners | Non partners | Mean views | Median views |\n'
        result += '|------|-------|------|-------|------|-------|\n'
        vectors = []

        for cluster_label, nodes in clusters.items():
            non_partners = 0
            partners = 0
            views = []

            for node in nodes:
                views.append(int(graph.nodes[node]['views']))
                if graph.nodes[node]['partner'] == 'False':
                    non_partners += 1
                elif graph.nodes[node]['partner'] == 'True':
                    partners += 1

            vectors.append(views)
            result += f'| {cluster_label} | {len(nodes)} | {partners} | {non_partners}'
            result += f'| {utils.format_number(np.mean(views))} | {utils.format_number(np.median(views))} |\n'

        ploting.plot_twitch_barchart(f'{output_folder}/barchart.jpg', clusters, graph)
        ploting.plot_log10_kde(f"{output_folder}/kde.jpg", vectors, 'views')
        return result

    elif dataset == 'cosponsorship':
        result = '| Cluster Number | Nodes | Democrats | Non democrats | Mean le_score | Median le_score |\n'
        result += '|------|-------|------|-------|------|-------|\n'
        vectors = []

        for cluster_label, nodes in clusters.items():
            le_score = []
            non_democrats = 0
            democrats = 0

            for node in nodes:
                le_score.append(float(graph.nodes[node]['le_score']))
                if graph.nodes[node]['democrat'] == '0':
                    non_democrats += 1
                elif graph.nodes[node]['democrat'] == '1':
                    democrats += 1

            vectors.append(le_score)
            result += f'| {cluster_label} | {len(nodes)} | {democrats} | {non_democrats}'
            result += f'| {round(np.mean(le_score),2)} | {round(np.median(le_score),2)} |\n'
        
        ploting.plot_cosponsorship_barchart(f'{output_folder}/barchart.jpg', clusters, graph)
        ploting.plot_kde(f"{output_folder}/kde.jpg", vectors, 'le_score')
        return result
    
    elif dataset == 'gs':
        result = '| Cluster Number | Nodes | Mean(citation) | Mean h-index | Mean g-index |\n'
        result += '|------|-------|------|------|------|\n'
        vectors_h_index = []
        vectors_g_index = []
        vectors_citation = []

        for cluster_label, nodes in clusters.items():
            citations = []
            hindex = []
            gindex = []

            for node in nodes:
                citations.append(float(graph.nodes[node]['total_number_of_citations']))
                hindex.append(float(graph.nodes[node]['h-index']))
                gindex.append(float(graph.nodes[node]['g-index']))

            vectors_h_index.append(hindex)
            vectors_g_index.append(gindex)
            vectors_citation.append(citations)

            result += f'| {cluster_label} | {utils.format_number(len(nodes))}'
            result += f'| {utils.format_number(np.mean(citations))}'
            result += f'| {round(np.mean(hindex),1)}'
            result += f'| {round(np.mean(gindex),1)}|\n'

        ploting.plot_kde(f"{output_folder}/hindex.jpg", vectors_h_index, 'h-index')
        ploting.plot_kde(f"{output_folder}/gindex.jpg", vectors_g_index, 'g-index')
        ploting.plot_log10_kde(f"{output_folder}/citation.jpg", vectors_citation, 'citation count')
        return result