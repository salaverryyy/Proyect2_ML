import numpy as np
from sklearn.cluster import DBSCAN

# Cargar datos reducidos
features_2d = np.load("data/reduction/test_tsne_2d.npy")

# Aplicar DBSCAN
dbscan = DBSCAN(eps=5, min_samples=5)  # Puedes ajustar eps y min_samples según los datos
labels = dbscan.fit_predict(features_2d)

# Guardar etiquetas
np.save("data/reduction/test_dbscan_labels.npy", labels)

print("✅ Clustering con DBSCAN completado y guardado.")
