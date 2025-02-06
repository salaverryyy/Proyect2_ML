import numpy as np
import os
from sklearn.manifold import TSNE

# Directorios
input_path = "data/r2plus1d_18_16_kinetics" # Carpeta donde están los archivos .npy
output_path = "data/reduction" # Carpeta donde se guardarán los resultados

# Crear la carpeta de salida
os.makedirs(output_path, exist_ok=True)

# Cargar los features extraídos
features_path = os.path.join(input_path, "features_test.npy")  # Ajusta el nombre del archivo si es diferente
features = np.load(features_path, allow_pickle=True)

# Reducir la dimensionalidad a 2D con t-SNE
tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
features_tsne_2d = tsne_2d.fit_transform(features)

# Reducir la dimensionalidad a 3D con t-SNE
tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30)
features_tsne_3d = tsne_3d.fit_transform(features)

# Guardar los resultados
np.save(os.path.join(output_path, "test_tsne_2d.npy"), features_tsne_2d)
np.save(os.path.join(output_path, "test_tsne_3d.npy"), features_tsne_3d)

print("✅ Reducción con t-SNE completada y guardada en", output_path)
