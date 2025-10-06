import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from src import pipeline
import pandas as pd

FOLDER_PATH = 'monoplex/choosing_k'

def for_dataset(dataset):
    types = ['weighted']

    for type in types:
        print(f"Determining best k for dataset '{dataset}' with type '{type}'")
        determine_number_of_clusters(dataset, type)

def determine_number_of_clusters(dataset, type, n_simulations=10_000, k_max=10):
    IA,time_per_simulation,average_weight = pipeline.run(dataset, type, n_simulations)
    print("Shape of Information Access Matrix:", IA.shape)
    IA = pipeline.normalize_offdiag(IA)
    print("Shape of Normalized Information Access Matrix:", IA.shape)

    silhouette_scores = []
    inertias = []
    k_range_elbow = range(1, k_max + 1)
    k_range_silhouette = range(2, k_max + 1)

    for k in k_range_elbow:
        print(f"Calculating inertia for k={k}")
        kmeans = KMeans(n_clusters=k, random_state=1, n_init='auto')
        kmeans.fit(IA)
        inertias.append(kmeans.inertia_ / IA.shape[0])

    for k in k_range_silhouette:
        print(f"Calculating silhouette score for k={k}")
        kmeans = KMeans(n_clusters=k, random_state=1, n_init='auto')
        labels = kmeans.fit_predict(IA)
        score = silhouette_score(IA, labels)
        silhouette_scores.append(round(score, 3))

    print("Inertias:", inertias)

    plt.figure()
    plt.plot(list(k_range_elbow), inertias, marker='o')
    plt.xticks(list(k_range_elbow))
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion')
    plt.title(f'Elbow method for IAC Google Scholar {type}')
    plt.grid(False)
    plt.savefig(f'{FOLDER_PATH}/{dataset}/elbow_{dataset}_{type}.png')

    print("Silhouette Scores:", silhouette_scores)

    silhouette_df = pd.DataFrame({
        'k': list(k_range_silhouette),
        'Silhouette Score': silhouette_scores
    })
    silhouette_df.to_csv(f'{FOLDER_PATH}/{dataset}/silhouette_{dataset}_{type}.csv', index=False)
