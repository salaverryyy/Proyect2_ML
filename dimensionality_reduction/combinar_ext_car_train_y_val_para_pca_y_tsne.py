import numpy as np
import os

# Función para combinar características de una carpeta y guardar en un solo archivo .npy
def combine_features(data_dir, output_filename):
    # Obtener lista de archivos .npy
    npy_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]

    if not npy_files:
        print(f"No se encontraron archivos .npy en {data_dir}")
        return

    # Cargar y combinar las características
    features_list = [np.load(os.path.join(data_dir, f)) for f in npy_files]

    # Concatenar todas las características
    features_combined = np.vstack(features_list)  # Si las matrices tienen la misma forma en la segunda dimensión

    # Guardar el archivo combinado
    np.save(os.path.join(data_dir, output_filename), features_combined)
    print(f" Archivo {output_filename} creado en {data_dir}")

# Carpetas de los conjuntos de datos
combine_features("data/r2plus1d_18_16_kinetics_train", "features_train.npy")
combine_features("data/r2plus1d_18_16_kinetics_val", "features_val.npy")

print(" Archivos combinados creados para train y val.")
