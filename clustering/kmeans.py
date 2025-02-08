import numpy as np
import os
from sklearn.cluster import KMeans

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

# Número de clusters para K-Means
K = 5

for file in datasets:
    print(f"🔹 Procesando clustering con K-Means para {file}...")
    
    # Cargar los datos reducidos
    features = np.load(os.path.join(input_path, file))

    # Aplicar K-Means
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    # Guardar las etiquetas
    output_file = file.replace(".npy", "_kmeans_labels.npy")
    np.save(os.path.join(output_path, output_file), labels)

    print(f" Clustering con K-Means completado para {file}. Guardado en {output_file}")

print(" Proceso de clustering con K-Means finalizado.")

'''
K-Means se usa porque es un método particional que asume que los clusters tienen una forma esférica, lo cual es ideal cuando los datos son compactos y balanceados.
'''
