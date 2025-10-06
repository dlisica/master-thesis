import os
import numpy as np
import json

from src import pipeline
from src import reporting
OUTPUT_PATH = 'monoplex/output'

def main():
    dataset = "cosponsorship"  # options: gs, cosponsorship, twitch, flickr_friendship, flickr_tag_similarity
    type = "weighted" # options: ndlib, random, uniform, weighted, trivalency
    n_simulations = 10_000
    n_clusters = 2

    #create output subfolder
    output_folder = f'{OUTPUT_PATH}/{dataset}_{type}_{n_simulations}_{n_clusters}'
    os.makedirs(output_folder, exist_ok=True)

    #simulator
    IA,time_per_simulation,average_weight = pipeline.run(dataset, type, n_simulations)
    np.save(f"{output_folder}/IA.npy", IA)

    #clustering
    clusters = pipeline.clustering(n_clusters, IA)
    with open(f'{output_folder}/clusters.json', 'w') as file:
        json.dump(clusters, file, indent=4)

    print("Time per simulation:", time_per_simulation)
    print("Average edge weight:", average_weight)

    #plots and analysis
    reporting.create_report(
        output_folder,
        clusters,
        dataset,
        time_per_simulation,
        average_weight
    )
    
if __name__ == "__main__":
    main()