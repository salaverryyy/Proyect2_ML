import numpy as np
import os
from sklearn.decomposition import PCA

# Directorios
input_paths = {
    "train": "data/r2plus1d_18_16_kinetics_train",
    "val": "data/r2plus1d_18_16_kinetics_val",
    "test": "data/r2plus1d_18_16_kinetics"
}
output_path = "data/reduction"  # Carpeta donde se guardarán los resultados

# Crear la carpeta de salida si no existe
os.makedirs(output_path, exist_ok=True)

# Archivos de entrada
datasets = {
    "train": "features_train.npy",
    "val": "features_val.npy",
    "test": "features_test.npy"
}

# Aplicar PCA a cada dataset
for name, filename in datasets.items():
    features_path = os.path.join(input_paths[name], filename)

    if not os.path.exists(features_path):
        print(f" Archivo no encontrado: {features_path}, saltando este conjunto.")
        continue

    # Cargar las características
    features = np.load(features_path, allow_pickle=True)

    # Reducir a 2D con PCA
    pca_2d = PCA(n_components=2)
    features_pca_2d = pca_2d.fit_transform(features)

    # Reducir a 3D con PCA
    pca_3d = PCA(n_components=3)
    features_pca_3d = pca_3d.fit_transform(features)

    # Guardar los resultados
    np.save(os.path.join(output_path, f"{name}_pca_2d.npy"), features_pca_2d)
    np.save(os.path.join(output_path, f"{name}_pca_3d.npy"), features_pca_3d)

    print(f"PCA aplicado y guardado para {name}")

print("Reducción con PCA completada para todos los conjuntos de datos.")
