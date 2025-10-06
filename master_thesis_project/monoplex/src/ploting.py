import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def plot_kde(path, vectors, value_name):
    for i in range(len(vectors)):
        sns.kdeplot(vectors[i], fill=False, label=f'cluster {i}')

    plt.title(f'{value_name} density')
    plt.xlabel(f'{value_name}')
    plt.legend(frameon=False)   
    plt.savefig(path)
    plt.close()
    
def plot_log10_kde(path, vectors, value_name):
    for i in range(len(vectors)):
        log_vectors = np.log10(vectors[i])
        sns.kdeplot(log_vectors, fill=False, label=f'cluster {i}')

    plt.title(f'{value_name} density')
    plt.xlabel(rf'$\log_{{10}}$({value_name})')
    plt.legend(frameon=False)
    plt.savefig(path)
    plt.close()

### Cosponsorship ###

def plot_cosponsorship_barchart(path, clusters, graph):
    cluster_ids = []
    democrat_ratios = []
    not_democrat_ratios = []

    for cluster, node_ids in clusters.items():
        total_nodes = len(node_ids)
        democrats = sum(graph.nodes[node]['democrat'] == '1' for node in node_ids)
        not_democrats = total_nodes - democrats
        cluster_ids.append(cluster)
        democrat_ratios.append(democrats / total_nodes)
        not_democrat_ratios.append(not_democrats / total_nodes)

    # Bar chart setup
    x = np.arange(len(cluster_ids))
    bar_width = 0.8

    fig, ax = plt.subplots()
    ax.bar(x, not_democrat_ratios, bar_width, label='non-democrat', color='steelblue')
    ax.bar(x, democrat_ratios, bar_width, bottom=not_democrat_ratios, label='democrat', color='orchid')

    # Labels and legend
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Probability')
    ax.set_title("Frequency of 'democrat' and 'non-democrat' across clusters")
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_ids)
    ax.legend(loc='upper right')

    plt.savefig(path)
    plt.close()

### Twitch ###

def plot_twitch_barchart(path, clusters, graph):
    cluster_ids = []
    democrat_ratios = []
    not_democrat_ratios = []

    for cluster, node_ids in clusters.items():
        total_nodes = len(node_ids)
        democrats = sum(graph.nodes[node]['partner'] == 'True' for node in node_ids)
        not_democrats = total_nodes - democrats
        cluster_ids.append(cluster)
        democrat_ratios.append(democrats / total_nodes)
        not_democrat_ratios.append(not_democrats / total_nodes)

    # Bar chart setup
    x = np.arange(len(cluster_ids))
    bar_width = 0.8

    fig, ax = plt.subplots()
    ax.bar(x, not_democrat_ratios, bar_width, label='non_partner', color='steelblue')
    ax.bar(x, democrat_ratios, bar_width, bottom=not_democrat_ratios, label='partner', color='orchid')

    # Labels and legend
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Probability')
    ax.set_title("Frequency of 'partner' and 'non-partner' across clusters")
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_ids)
    ax.legend(loc='upper right')

    plt.savefig(path)
    plt.close()