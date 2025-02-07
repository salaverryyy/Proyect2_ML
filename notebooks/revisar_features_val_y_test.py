import numpy as np
import os

# Rutas de los archivos
features_val_path = "data/r2plus1d_18_16_kinetics_val/features_val.npy"
features_test_path = "data/r2plus1d_18_16_kinetics/features_test.npy"  # Ajusta la ruta si es diferente

# Cargar y mostrar información de features_val.npy
if os.path.exists(features_val_path):
    features_val = np.load(features_val_path)
    print("Shape de features_val:", features_val.shape)
    print("Primeras filas de features_val:\n", features_val[:5])

# Cargar y mostrar información de features_test.npy
if os.path.exists(features_test_path):
    features_test = np.load(features_test_path)
    print("\nShape de features_test:", features_test.shape)
    print("Primeras filas de features_test:\n", features_test[:5])
