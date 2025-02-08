import numpy as np
import os
from sklearn.cluster import DBSCAN

# Directorio donde están los archivos reducidos
input_path = "data/reduction"
output_path = "data/reduction"

# Listado de archivos que contienen datos reducidos
datasets = [
    "test_pca_2d.npy", "test_pca_3d.npy",
    "train_pca_2d.npy", "train_pca_3d.npy",
    "val_pca_2d.npy", "val_pca_3d.npy",
    "test_tsne_2d.npy", "test_tsne_3d.npy",
    "train_tsne_2d.npy", "train_tsne_3d.npy",
    "val_tsne_2d.npy", "val_tsne_3d.npy"
]

# Parámetros para DBSCAN
EPS = 5  # Radio de vecindad
MIN_SAMPLES = 5  # Cantidad mínima de vecinos

for file in datasets:
    print(f"🔹 Procesando clustering con DBSCAN para {file}...")
    
    # Cargar los datos reducidos
    features = np.load(os.path.join(input_path, file))

    # Aplicar DBSCAN
    dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
    labels = dbscan.fit_predict(features)

    # Guardar las etiquetas
    output_file = file.replace(".npy", "_dbscan_labels.npy")
    np.save(os.path.join(output_path, output_file), labels)

    print(f"Clustering con DBSCAN completado para {file}. Guardado en {output_file}")

print(" Proceso de clustering con DBSCAN finalizado.")

'''
DBSCAN es un método basado en densidad que identifica clusters de diferentes formas y puede detectar ruido. Es útil para descubrir estructuras complejas en los datos.
'''