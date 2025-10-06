import json
import itertools
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pandas as pd

def load_cluster_labels(path):
    with open(path, 'r') as f:
        clusters = json.load(f)

    labels = {}
    for cluster_id, node_list in clusters.items():
        for node in node_list:
            labels[int(node)] = int(cluster_id)
    return labels

def compute_alignment_matrix(cluster_paths, method_names):
    label_dicts = [load_cluster_labels(path) for path in cluster_paths]
    all_nodes = set.intersection(*[set(d.keys()) for d in label_dicts])
    sorted_nodes = sorted(all_nodes)
    label_lists = [[labels[n] for n in sorted_nodes] for labels in label_dicts]

    n = len(label_lists)
    ari_matrix = pd.DataFrame(index=method_names, columns=method_names)
    nmi_matrix = pd.DataFrame(index=method_names, columns=method_names)

    for i, j in itertools.product(range(n), repeat=2):
        ari = adjusted_rand_score(label_lists[i], label_lists[j])
        nmi = normalized_mutual_info_score(label_lists[i], label_lists[j])
        ari_matrix.iloc[i, j] = round(ari, 4)
        nmi_matrix.iloc[i, j] = round(nmi, 4)

    return ari_matrix, nmi_matrix

# change accordingly
cluster_paths = [
    "monoplex/output/cosponsorship_weighted_10000_2/clusters.json",
    "monoplex/output/cosponsorship_random_10000_2/clusters.json",
    "monoplex/output/cosponsorship_trivalency_10000_2/clusters.json",
    "monoplex/output/cosponsorship_uniform_10000_2/clusters.json",
    "monoplex/output/cosponsorship_ndlib_10000_2/clusters.json",
]

method_names = [
    "weighted",
    "random",
    "trivalency",
    "uniform",
    "ndlib"
]

ari_table, nmi_table = compute_alignment_matrix(cluster_paths, method_names)

print("NMI Matrix")
print(nmi_table)
print("\nARI Matrix")
print(ari_table)
