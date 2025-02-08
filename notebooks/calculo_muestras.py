import numpy as np

# Cargar los features usados en clustering
test_features = np.load("data/reduction/test_tsne_2d.npy")
print(f"Total de instancias en test_tsne_2d.npy: {test_features.shape[0]}")

test_features_pca = np.load("data/reduction/test_pca_2d.npy")
print(f"Total de instancias en test_pca_2d.npy: {test_features_pca.shape[0]}")
