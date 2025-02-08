import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
import os

# Directorios de clustering
clustering_dirs = {
    "dbscan": "data/dbscan_clustering",
    "kmeans": "data/kmeans_clustering"
}

# Directorio de los features reducidos
features_dir = "data/reduction"

# Evaluar cada método de clustering
for method, path in clustering_dirs.items():
    print(f"\nEvaluando {method.upper()}...\n")

    # Listar los archivos con etiquetas de clustering
    label_files = sorted([f for f in os.listdir(path) if f.endswith("_labels.npy")])

    for file in label_files:
        feature_file = file.replace(f"_{method}_labels.npy", ".npy")  # Obtener el archivo de features original
        feature_path = os.path.join(features_dir, feature_file)

        if not os.path.exists(feature_path):
            print(f"Archivo de features no encontrado para {file}.")
            continue

        labels = np.load(os.path.join(path, file))
        features = np.load(feature_path)

        if len(set(labels)) > 1:  # Evitar errores si solo hay un cluster
            silhouette = silhouette_score(features, labels)
            rand_index = adjusted_rand_score(labels, labels)  # No hay ground truth, se evalúa sobre sí mismo
            mutual_info = adjusted_mutual_info_score(labels, labels)

            print(f"{file}:")
            print(f"  - Silhouette Score: {silhouette:.4f}")
            print(f"  - Rand Index: {rand_index:.4f}")
            print(f"  - Mutual Information: {mutual_info:.4f}\n")
        else:
            print(f"{file} tiene un solo cluster, no se puede calcular Silhouette Score.\n")
