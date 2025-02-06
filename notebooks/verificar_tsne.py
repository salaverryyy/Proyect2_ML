import numpy as np

# Cargar los datos reducidos
features_2d = np.load("data/reduction/test_tsne_2d.npy")
features_3d = np.load("data/reduction/test_tsne_3d.npy")

print("Forma de test_tsne_2d:", features_2d.shape)  # Debe ser (num_samples, 2)
print("Forma de test_tsne_3d:", features_3d.shape)  # Debe ser (num_samples, 3))
