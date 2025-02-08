import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Cargar datos reducidos
features_2d = np.load("data/reduction/test_tsne_2d.npy")

# Probar diferentes valores de eps y min_samples
eps_values = [2, 3, 5, 10]
min_samples_values = [3, 5, 10]

best_eps = None
best_min_samples = None
best_score = -1
best_labels = None

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features_2d)

        # Verificar si se crearon múltiples clusters
        if len(set(labels)) > 1:
            silhouette = silhouette_score(features_2d, labels)
            print(f"DBSCAN (eps={eps}, min_samples={min_samples}): Silhouette Score = {silhouette:.4f}")

            # Guardar los mejores parámetros
            if silhouette > best_score:
                best_score = silhouette
                best_eps = eps
                best_min_samples = min_samples
                best_labels = labels

# Guardar el mejor resultado
if best_labels is not None:
    np.save("data/reduction/test_dbscan_labels.npy", best_labels)
    print(f"Mejor configuración: eps={best_eps}, min_samples={best_min_samples}, Silhouette Score={best_score:.4f}")
else:
    print("DBSCAN no encontró una buena separación de clusters.")
