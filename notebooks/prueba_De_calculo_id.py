import numpy as np

test_labels_file = "data/kmeans_clustering/test_pca_2d_kmeans_labels.npy"
test_ids_file = "data/r2plus1d_18_16_kinetics/path_test.txt"

# Cargar datos
labels = np.load(test_labels_file)
with open(test_ids_file, "r") as f:
    youtube_ids = [line.strip() for line in f.readlines()]

# Comparar tamaños
print(f"Total de IDs en path_test.txt: {len(youtube_ids)}")
print(f"Total de etiquetas en test_pca_2d_kmeans_labels.npy: {len(labels)}")
