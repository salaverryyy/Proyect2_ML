import numpy as np
import os

# Carpeta donde están los archivos
data_dir = "data/r2plus1d_18_16_kinetics"

# Obtener lista de archivos .npy
npy_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]

# Cargar y combinar las características
features_list = [np.load(os.path.join(data_dir, f)) for f in npy_files]

# Concatenar todas las características
features_combined = np.vstack(features_list)  # Si las matrices tienen la misma forma en la segunda dimensión

# Guardar el archivo combinado
np.save(os.path.join(data_dir, "features_test.npy"), features_combined)

print("Archivo features_test.npy creado con éxito.")
