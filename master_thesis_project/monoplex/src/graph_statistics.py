import utils
import numpy as np
import matplotlib.pyplot as plt

### Google Scholar ###

def gs_statistics():
    print('google-scholar graph statistics:')
    print()

    G = utils.load_graph('gs')
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    print(f'is_directed: {G.is_directed()}')
    print(f'#nodes: {n_nodes}')
    print(f'#edges: {n_edges}')

    max_edges = n_nodes * (n_nodes - 1)
    density = 2 * n_edges / max_edges if max_edges > 0 else 0
    print(f'density: {round(density, 4)}')

    citations = [float(G.nodes()[node]['total_number_of_citations']) for node in G.nodes()]
    print()
    print(f'citations min: {round(min(citations))}')
    print(f'citations max: {round(max(citations))}')
    print(f'citations mean: {round(np.mean(citations))}')

    h_index = [float(G.nodes()[node]['h-index']) for node in G.nodes()]
    print()
    print(f'h_index min: {round(min(h_index))}')
    print(f'h_index max: {round(max(h_index))}')
    print(f'h_index mean: {round(np.mean(h_index))}')

    g_index = [float(G.nodes()[node]['g-index']) for node in G.nodes()]
    print()
    print(f'g_index min: {round(min(g_index))}')
    print(f'g_index max: {round(max(g_index))}')
    print(f'g_index mean: {round(np.mean(g_index))}')

    plot_citations(citations)
    plot_h_index(h_index)
    #plot_g_index(g_index)

def plot_citations(citations):
    log_citations = np.log10(citations)

    bins = np.arange(0, 7, 1)
    counts, _ = np.histogram(log_citations, bins=bins)
    plt.bar(bins[:-1], counts, width=1, align='edge', edgecolor='black')
    plt.title('Distribution of log(citation count)')
    plt.xlabel('log(citation count)')
    plt.ylabel('count')
    plt.xticks(bins[:-1])
    plt.show()

def plot_h_index(h_index):
    bins = np.arange(0, 100, 5)
    counts, _ = np.histogram(h_index, bins=bins)
    plt.bar(bins[:-1], counts, width=5, align='edge', edgecolor='black')
    plt.title('Distribution of h-index')
    plt.xlabel('h-index')
    plt.ylabel('count')
    plt.xticks(bins[:-1])
    plt.show()

def plot_g_index(g_index):
    bins = np.arange(0, 100, 5)
    counts, _ = np.histogram(g_index, bins=bins)
    plt.bar(bins[:-1], counts, width=5, align='edge', edgecolor='black')
    plt.title('Distribution of g_index')
    plt.xlabel('g_index')
    plt.ylabel('count')
    plt.xticks(bins[:-1])
    plt.show()

### Cosponsorship ###

def cosponsorship_statistics():
    print('Co-sponsorship graph statistics:')
    print()

    G = utils.load_graph('cosponsorship')
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    print(f'is_directed: {G.is_directed()}')
    print(f'#nodes: {n_nodes}')
    print(f'#edges: {n_edges}')

    max_edges = n_nodes * (n_nodes - 1)
    density = n_edges / max_edges if max_edges > 0 else 0
    print(f'density: {round(density, 4)}')

    le_scores = [float(G.nodes()[node]['le_score']) for node in G.nodes()]
    print()
    print(f'le_score min: {round(min(le_scores), 4)}')
    print(f'le_score max: {round(max(le_scores), 4)}')
    print(f'le_score mean: {round(np.mean(le_scores), 4)}')

    democrats = [node for node in G.nodes() if G.nodes()[node]['democrat'] == '1']
    d = len(democrats)
    nd = n_nodes - len(democrats)

    print()
    print(f'#democrats: {d} {round(d * 100 / n_nodes, 1)}%')
    print(f'#non_democrats: {nd} {round(nd * 100 / n_nodes, 1)}%')

    plot_le_score(le_scores)

def plot_le_score(le_scores):
    bins = np.arange(0, 10, 1)
    counts, _ = np.histogram(le_scores, bins=bins)
    plt.bar(bins[:-1], counts, width=1, align='edge', edgecolor='black')
    plt.title('Distribution of le_score')
    plt.xlabel('le_score')
    plt.ylabel('count')
    plt.xticks(bins[:-1])
    plt.show()

### Twitch ###

def twitch_statistics():
    print('Twitch graph statistics:')
    print()

    G = utils.load_graph('twitch')
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    print(f'is_directed: {G.is_directed()}')
    print(f'#nodes: {n_nodes}')
    print(f'#edges: {n_edges}')

    max_edges = n_nodes * (n_nodes - 1)
    density = 2 * n_edges / max_edges if max_edges > 0 else 0
    print(f'density: {round(density, 4)}')

    views = [float(G.nodes()[node]['views']) for node in G.nodes()]
    print()
    print(f'views min: {round(min(views))}')
    print(f'views max: {round(max(views))}')
    print(f'views mean: {round(np.mean(views))}')

    partners = [node for node in G.nodes() if G.nodes()[node]['partner'] == 'True']
    p = len(partners)
    non_p = n_nodes - len(partners)

    print()
    print(f'#partners: {p} {round(p * 100 / n_nodes, 1)}%')
    print(f'#non_partners: {non_p} {round(non_p * 100 / n_nodes, 1)}%')

    plot_views(views)

def plot_views(views):
    log_views = np.log10(views)

    bins = np.arange(0, 10, 1)
    counts, _ = np.histogram(log_views, bins=bins)
    plt.bar(bins[:-1], counts, width=1, align='edge', edgecolor='black')
    plt.title('Distribution of log(views)')
    plt.xlabel('log(views)')
    plt.ylabel('count')
    plt.xticks(bins[:-1])
    plt.show()

if __name__ == '__main__':
    #gs_statistics()
    # cosponsorship_statistics()
    twitch_statistics()