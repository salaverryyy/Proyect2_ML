import numpy as np
import matplotlib.pyplot as plt

def region_query(X, point_idx, eps):
    # Calculamos las distancias Euclidianas desde X[point_idx] hasta todos los puntos de X
    point = X[point_idx]
    distances = np.linalg.norm(X - point, axis=1)
    return list(np.where(distances <= eps)[0])

def expand_cluster(X, labels, point_idx, neighbors, cluster_id, eps, min_samples, visited):
    # Asignamos el cluster actual al punto
    labels[point_idx] = cluster_id
    # Usamos una lista (como cola) para ir recorriendo los vecinos
    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]
        # Si el vecino no ha sido visitado, lo marcamos y revisamos sus vecinos
        if not visited[neighbor_idx]:
            visited[neighbor_idx] = True
            neighbor_neighbors = region_query(X, neighbor_idx, eps)
            # Si el vecino tiene suficientes puntos en su vecindario, lo consideramos núcleo y ampliamos la lista
            if len(neighbor_neighbors) >= min_samples:
                # Agregamos aquellos puntos que aún no estén en la lista de vecinos
                for n in neighbor_neighbors:
                    if n not in neighbors:
                        neighbors.append(n)
        # Si el vecino no ha sido asignado a ningún cluster (incluyendo si estaba marcado como ruido)
        if labels[neighbor_idx] == -1:
            labels[neighbor_idx] = cluster_id
        i += 1

def dbscan(X, eps, min_samples):
    n_points = X.shape[0]
    labels = np.full(n_points, -1)
    visited = np.zeros(n_points, dtype=bool)
    cluster_id = 0
    
    for point_idx in range(n_points):
        if visited[point_idx]:
            continue
        visited[point_idx] = True
        neighbors = region_query(X, point_idx, eps)
        if len(neighbors) < min_samples:
            labels[point_idx] = -1
        else:
            expand_cluster(X, labels, point_idx, neighbors, cluster_id, eps, min_samples, visited)
            cluster_id += 1
    return labels

# ------------------- Uso del algoritmo DBSCAN desde cero -------------------

# Cargar datos reducidos
features_2d = np.load("data/reduction/test_tsne_2d.npy")

# Parámetros para DBSCAN
eps = 5
min_samples = 5

labels = dbscan(features_2d, eps, min_samples)

np.save("data/reduction/test_dbscan_labels.npy", labels)

print("✅ Clustering con DBSCAN implementado desde cero completado y guardado.")

"""
# Visualización de los clusters

plt.figure(figsize=(8, 6))
unique_labels = np.unique(labels)
for label in unique_labels:
    if label == -1:
        plt.scatter(features_2d[labels == label, 0],
                    features_2d[labels == label, 1],
                    c="gray", label="Ruido", marker="x")
    else:
        plt.scatter(features_2d[labels == label, 0],
                    features_2d[labels == label, 1],
                    label=f"Cluster {label}")
plt.title("Clustering con DBSCAN (implementación desde cero)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.legend()
plt.show()
"""