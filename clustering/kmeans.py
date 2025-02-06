import numpy as np
from sklearn.cluster import KMeans

# Cargar datos reducidos
features_2d = np.load("data/reduction/test_tsne_2d.npy")

# Aplicar K-Means con un número de clusters K=5 (ajústalo según sea necesario)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = kmeans.fit_predict(features_2d)

# Guardar etiquetas
np.save("data/reduction/test_kmeans_labels.npy", labels)

print("✅ Clustering con K-Means completado y guardado.")
