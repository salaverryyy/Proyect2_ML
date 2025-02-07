import numpy as np
import os

# Ruta corregida del archivo (ajusta la extensión si es necesario)
features_train_path = "data/r2plus1d_18_16_kinetics_train/features_train.npy"

# Cargar el archivo (si es .npy)
features_train = np.load(features_train_path)

# Mostrar dimensiones del array
print("Shape de features_train:", features_train.shape)

# Mostrar los primeros datos
print("Primeras filas de features_train:")
print(features_train[:5])

# Verificar cuántos archivos hay en la carpeta
train_folder = "data/r2plus1d_18_16_kinetics_train"
train_files = os.listdir(train_folder)
print("Número de archivos en la carpeta:", len(train_files))
print("Ejemplo de nombres de archivos:", train_files[:5])
