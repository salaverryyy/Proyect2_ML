import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

# Cargar datos y etiquetas
features_2d = np.load("data/reduction/test_tsne_2d.npy")
labels_kmeans = np.load("data/reduction/test_kmeans_labels.npy")
labels_dbscan = np.load("data/reduction/test_dbscan_labels.npy")

# Si tienes etiquetas reales, cárgalas aquí
# true_labels = np.load("data/reduction/true_labels.npy")  # Descomenta y ajusta si las tienes

# Función para calcular métricas
def evaluate_clustering(features, labels, method_name):
    unique_labels = set(labels)
    if len(unique_labels) > 1:  # Solo calculamos si hay más de un cluster
        silhouette = silhouette_score(features, labels)
        # rand_index = adjusted_rand_score(true_labels, labels)  # Si tienes true_labels
        # mutual_info = normalized_mutual_info_score(true_labels, labels)  # Si tienes true_labels
        print(f"📊 Métricas para {method_name}:")
        print(f"   - Silhouette Score: {silhouette:.4f}")
        # print(f"   - Rand Index: {rand_index:.4f}")
        # print(f"   - Mutual Information: {mutual_info:.4f}")
    else:
        print(f"⚠️ No se pudieron calcular métricas para {method_name} (posible único cluster o ruido).")

# Evaluar KMeans
evaluate_clustering(features_2d, labels_kmeans, "KMeans")

# Evaluar DBSCAN
evaluate_clustering(features_2d, labels_dbscan, "DBSCAN")
