import os
import numpy as np
import json

from src import pipeline
from src import utils

OUTPUT_PATH = 'multiplex/output'

def main(type, inter_layer_threshold):
    dataset = "politicsuk"  # Options: 'flickr', 'politicsuk'
    type = "weighted"  # Options: 'random', 'uniform', 'weighted', 'trivalency'
    n_simulations = 10_000
    n_clusters = 4
    inter_layer_threshold = 0.2

    #create output subfolder
    output_folder = f'{OUTPUT_PATH}/{dataset}_{type}_{n_simulations}_{inter_layer_threshold}'
    os.makedirs(output_folder, exist_ok=True)

    #simulator
    IA = pipeline.run(dataset, type, n_simulations, inter_layer_threshold)
    np.save(f"{output_folder}/IA.npy", IA)

    #clustering
    clusters = pipeline.clustering(n_clusters, IA)
    with open(f'{output_folder}/clusters.json', 'w') as file:
        json.dump(clusters, file, indent=4)

if __name__ == "__main__":
    main()