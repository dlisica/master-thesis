import json
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from utils import load_ground_truth, get_predicted_labels

def score_against_ground_truth(path, ground_truth):
    try:
        predicted_labels = get_predicted_labels(json.load(open(path, 'r')))

        ari = round(adjusted_rand_score(ground_truth, predicted_labels), 4)
        nmi = round(normalized_mutual_info_score(ground_truth, predicted_labels), 4)
        return nmi, ari
    except FileNotFoundError:
        return 'NA', 'NA'

def compare_flickr():
    ground_truth = load_ground_truth('flickr')

    graph_names = ["Multiplex-OR", "Multiplex-AND", "Friendship", "Tag-similarity"]
    method_names = ["Weighted", "Random", "Trivalency", "Uniform"]

    paths = {
        "Multiplex-OR": {
            "Weighted": "multiplex/output/flickr_weighted_10000_0.2/clusters.json",
            "Random": "multiplex/output/flickr_random_10000_0.2/clusters.json",
            "Trivalency": "multiplex/output/flickr_trivalency_10000_0.2/clusters.json",
            "Uniform": "multiplex/output/flickr_uniform_10000_0.2/clusters.json"
        },
        "Multiplex-AND": {
            "Weighted": "multiplex/output/flickr_weighted_10000_0.8/clusters.json",
            "Random": "multiplex/output/flickr_random_10000_0.8/clusters.json",
            "Trivalency": "multiplex/output/flickr_trivalency_10000_0.8/clusters.json",
            "Uniform": "multiplex/output/flickr_uniform_10000_0.8/clusters.json"
        },
        "Friendship": {
            "Weighted": "monoplex/output/flickr_friendship_weighted_10000_7/clusters.json",
            "Random": "monoplex/output/flickr_friendship_random_10000_7/clusters.json",
            "Trivalency": "monoplex/output/flickr_friendship_trivalency_10000_7/clusters.json",
            "Uniform": "monoplex/output/flickr_friendship_uniform_10000_7/clusters.json"
        },
        "Tag-similarity": {
            "Weighted": "monoplex/output/flickr_tag_similarity_weighted_10000_7/clusters.json",
            "Random": "monoplex/output/flickr_tag_similarity_random_10000_7/clusters.json",
            "Trivalency": "monoplex/output/flickr_tag_similarity_trivalency_10000_7/clusters.json",
            "Uniform": "monoplex/output/flickr_tag_similarity_uniform_10000_7/clusters.json"
        }
    }

    nmi_table = pd.DataFrame(index=graph_names, columns=method_names)
    ari_table = pd.DataFrame(index=graph_names, columns=method_names)

    for graph in graph_names:
        for method in method_names:
            path = paths[graph][method]
            if path is None:
                nmi, ari = 'NA', 'NA'
            else:
                nmi, ari = score_against_ground_truth(path, ground_truth)
            nmi_table.loc[graph, method] = nmi
            ari_table.loc[graph, method] = ari

    print("Flickr NMI")
    print(nmi_table)
    print("\nFlickr ARI")
    print(ari_table)

def compare_politicsuk():
    ground_truth = load_ground_truth('politicsuk')

    graph_names = ["Multiplex-1", "Multiplex-2", "Multiplex-3", "Follows", "Mentions", "Retweets"]
    method_names = ["Weighted", "Random", "Trivalency", "Uniform"]

    paths = {
        "Multiplex-1": {
            "Weighted": "multiplex/output/politicsuk_weighted_10000_0.2/clusters.json",
            "Random": "multiplex/output/politicsuk_random_10000_0.2/clusters.json",
            "Trivalency": "multiplex/output/politicsuk_trivalency_10000_0.2/clusters.json",
            "Uniform": "multiplex/output/politicsuk_uniform_10000_0.2/clusters.json"
        },
        "Multiplex-2": {
            "Weighted": "multiplex/output/politicsuk_weighted_10000_0.5/clusters.json",
            "Random": "multiplex/output/politicsuk_random_10000_0.5/clusters.json",
            "Trivalency": "multiplex/output/politicsuk_trivalency_10000_0.5/clusters.json",
            "Uniform": "multiplex/output/politicsuk_uniform_10000_0.5/clusters.json"
        },
        "Multiplex-3": {
            "Weighted": "multiplex/output/politicsuk_weighted_10000_0.8/clusters.json",
            "Random": "multiplex/output/politicsuk_random_10000_0.8/clusters.json",
            "Trivalency": "multiplex/output/politicsuk_trivalency_10000_0.8/clusters.json",
            "Uniform": "multiplex/output/politicsuk_uniform_10000_0.8/clusters.json"
        },
        "Follows": {
            "Weighted": "monoplex/output/politicsuk_follows_weighted_10000_4/clusters.json",
            "Random": "monoplex/output/politicsuk_follows_random_10000_4/clusters.json",
            "Trivalency": "monoplex/output/politicsuk_follows_trivalency_10000_4/clusters.json",
            "Uniform": "monoplex/output/politicsuk_follows_uniform_10000_4/clusters.json"
        },
        "Mentions": {
            "Weighted": "monoplex/output/politicsuk_mentions_weighted_10000_4/clusters.json",
            "Random": "monoplex/output/politicsuk_mentions_random_10000_4/clusters.json",
            "Trivalency": "monoplex/output/politicsuk_mentions_trivalency_10000_4/clusters.json",
            "Uniform": "monoplex/output/politicsuk_mentions_uniform_10000_4/clusters.json"
        },
        "Retweets": {
            "Weighted": "monoplex/output/politicsuk_retweets_weighted_10000_4/clusters.json",
            "Random": "monoplex/output/politicsuk_retweets_random_10000_4/clusters.json",
            "Trivalency": "monoplex/output/politicsuk_retweets_trivalency_10000_4/clusters.json",
            "Uniform": "monoplex/output/politicsuk_retweets_uniform_10000_4/clusters.json"
        }
    }

    nmi_table = pd.DataFrame(index=graph_names, columns=method_names)
    ari_table = pd.DataFrame(index=graph_names, columns=method_names)

    for graph in graph_names:
        for method in method_names:
            path = paths[graph][method]
            if path is None:
                nmi, ari = 'NA', 'NA'
            else:
                nmi, ari = score_against_ground_truth(path, ground_truth)
            nmi_table.loc[graph, method] = nmi
            ari_table.loc[graph, method] = ari

    print("Politics UK NMI")
    print(nmi_table)
    print("\nPolitics UK ARI")
    print(ari_table)