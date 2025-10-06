import pandas as pd
from scipy.stats import kruskal, chi2_contingency, fisher_exact
import json
import numpy as np

import utils
import clustering_methods

def json_to_array(json):
    max_node_id = max(node for nodes in json.values() for node in nodes)
    labels = [-1] * (max_node_id + 1)

    for cluster_id, nodes in json.items():
        for node in nodes:
            labels[node] = int(cluster_id)

    return labels

### COSPONSORSHIP ###

def build_cluster_df_cosponsorship(labels):
    G = utils.load_graph("cosponsorship")
    records = []

    for index in range(len(labels)):
        node = G.nodes[index]
        
        records.append({
            "le_score": float(node['le_score']),
            "democrat": node['democrat'] == '1',
            "cluster": int(labels[index])
        })

    return pd.DataFrame(records)

def run_stat_tests_cosponsorship(df):
    results = {}

    groups = [df.loc[df['cluster'] == k, 'le_score'] for k in df['cluster'].unique()]
    stat, pval = kruskal(*groups)
    results['le_score'] = {"statistic": stat, "p_value": pval}

    contingency = pd.crosstab(df['democrat'], df['cluster'])
    if contingency.shape == (2, 2):
        _, pval = fisher_exact(contingency)
        results['democrat'] = {"test": "Fisher", "p_value": pval}
    else:
        chi2, pval, dof, _ = chi2_contingency(contingency)
        results['democrat'] = {"test": "Chi-square", "chi2": chi2, "dof": dof, "p_value": pval}

    return results

def values_cosponsorship():
    paths = [
        "monoplex/output/cosponsorship_weighted_10000_2/clusters.json",
        "monoplex/output/cosponsorship_random_10000_2/clusters.json",
        "monoplex/output/cosponsorship_trivalency_10000_2/clusters.json",
        "monoplex/output/cosponsorship_uniform_10000_2/clusters.json"
    ]

    for path in paths:
        json_file = json.load(open(path))
        array = json_to_array(json_file)
        df = build_cluster_df_cosponsorship(array)
        #print(df)
        results = run_stat_tests_cosponsorship(df)
        print("Kruskal–Wallis le_score:")
        print(results['le_score'])

        print("Test democrat:")
        print(results['democrat'])
        print()

    method_results = [
        clustering_methods.spectral_clustering("cosponsorship", k=2),
        clustering_methods.fc_clustering("cosponsorship", k=2),
        clustering_methods.louvain_clustering("cosponsorship"),
        clustering_methods.cp_clustering("cosponsorship")
    ]

    for array in method_results:
        if array is None:
            continue

        df = build_cluster_df_cosponsorship(array)
        #print(df)
        results = run_stat_tests_cosponsorship(df)
        print("Kruskal–Wallis le_score:")
        print(results['le_score'])

        print("Test democrat:")
        print(results['democrat'])
        print()

### TWITCH ###

def build_cluster_df_twitch(labels):
    G = utils.load_graph("twitch")
    records = []

    for index in range(len(labels)):
        node = G.nodes[index]
        
        records.append({
            "views": np.log10(float(node['views'])),
            "partner": node['partner'] == 'True',
            "cluster": int(labels[index])
        })

    return pd.DataFrame(records)

def run_stat_tests_twitch(df):
    results = {}

    groups = [df.loc[df['cluster'] == k, 'views'] for k in df['cluster'].unique()]
    stat, pval = kruskal(*groups)
    results['views'] = {"statistic": stat, "p_value": pval}

    contingency = pd.crosstab(df['partner'], df['cluster'])
    if contingency.shape == (2, 2):
        _, pval = fisher_exact(contingency)
        results['partner'] = {"test": "Fisher", "p_value": pval}
    else:
        chi2, pval, dof, _ = chi2_contingency(contingency)
        results['partner'] = {"test": "Chi-square", "chi2": chi2, "dof": dof, "p_value": pval}

    return results

def values_twitch():
    paths = [
        "monoplex/output/twitch_weighted_10000_2/clusters.json",
        "monoplex/output/twitch_random_10000_2/clusters.json",
        "monoplex/output/twitch_trivalency_10000_2/clusters.json",
        "monoplex/output/twitch_uniform_10000_2/clusters.json"
    ]

    for path in paths:
        json_file = json.load(open(path))
        array = json_to_array(json_file)
        df = build_cluster_df_twitch(array)
        #print(df)
        results = run_stat_tests_twitch(df)
        print("Kruskal–Wallis views:")
        print(results['views'])

        print("Test partner:")
        print(results['partner'])
        print()

    method_results = [
        clustering_methods.spectral_clustering("twitch", k=2),
        clustering_methods.fc_clustering("twitch", k=2),
        clustering_methods.louvain_clustering("twitch"),
        clustering_methods.cp_clustering("twitch")
    ]

    for array in method_results:
        if array is None:
            continue

        df = build_cluster_df_twitch(array)
        #print(df)
        results = run_stat_tests_twitch(df)
        print("Kruskal–Wallis views:")
        print(results['views'])

        print("Test partner:")
        print(results['partner'])
        print()

### GOOGLE SCHOLAR ###

def build_cluster_df_gs(labels):
    G = utils.load_graph("gs")
    records = []

    for index in range(len(labels)):
        node = G.nodes[index]
        
        records.append({
            "citation_count": np.log10(int(node['total_number_of_citations'])),
            "h_index": int(node['h-index']),
            "cluster": int(labels[index])
        })

    return pd.DataFrame(records)

def run_stat_tests_gs(df):
    results = {}

    groups = [df.loc[df['cluster'] == k, 'citation_count'] for k in df['cluster'].unique()]
    stat, pval = kruskal(*groups)
    results['citation_count'] = {"statistic": stat, "p_value": pval}

    groups = [df.loc[df['cluster'] == k, 'h_index'] for k in df['cluster'].unique()]
    stat, pval = kruskal(*groups)
    results['h_index'] = {"statistic": stat, "p_value": pval}

    return results

def values_gs():
    paths = [
        "monoplex/output/gs_weighted_10000_2/clusters.json",
        "monoplex/output/gs_random_10000_2/clusters.json",
        "monoplex/output/gs_trivalency_10000_2/clusters.json",
        "monoplex/output/gs_uniform_10000_2/clusters.json"
    ]

    for path in paths:
        json_file = json.load(open(path))
        array = json_to_array(json_file)
        df = build_cluster_df_gs(array)
        #print(df)
        results = run_stat_tests_gs(df)
        print("Kruskal–Wallis citation_count:")
        print(results['citation_count'])

        print("Kruskal–Wallis h_index:")
        print(results['h_index'])
        print()

    method_results = [
        clustering_methods.spectral_clustering("gs", k=2),
        clustering_methods.fc_clustering("gs", k=2),
        clustering_methods.louvain_clustering("gs"),
        clustering_methods.cp_clustering("gs")
    ]

    for array in method_results:
        if array is None:
            continue

        df = build_cluster_df_gs(array)
        #print(df)
        results = run_stat_tests_gs(df)
        print("Kruskal–Wallis citation_count:")
        print(results['citation_count'])

        print("Kruskal–Wallis h_index:")
        print(results['h_index'])
        print()