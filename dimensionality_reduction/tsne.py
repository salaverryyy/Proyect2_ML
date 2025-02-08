import numpy as np
import os
from sklearn.manifold import TSNE

# Directorio de salida
output_path = "data/reduction"
os.makedirs(output_path, exist_ok=True)

# Diccionario con los conjuntos de datos y sus rutas
datasets = {
    "train": "data/r2plus1d_18_16_kinetics_train/features_train.npy",
    "val": "data/r2plus1d_18_16_kinetics_val/features_val.npy"
}

# Aplicar t-SNE a cada conjunto de datos
for name, path in datasets.items():
    print(f"Procesando {name}...")
    features = np.load(path, allow_pickle=True)

    # t-SNE 2D
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
    features_tsne_2d = tsne_2d.fit_transform(features)
    np.save(os.path.join(output_path, f"{name}_tsne_2d.npy"), features_tsne_2d)

    # t-SNE 3D
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30)
    features_tsne_3d = tsne_3d.fit_transform(features)
    np.save(os.path.join(output_path, f"{name}_tsne_3d.npy"), features_tsne_3d)

    print(f"✅ Reducción con t-SNE completada para {name} y guardada en {output_path}")

print("🎯 Proceso finalizado.")
    