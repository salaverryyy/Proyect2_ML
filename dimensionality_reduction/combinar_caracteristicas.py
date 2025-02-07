import numpy as np
import os

def combine_features_v2(data_dir, output_filename):
    """
    Función para combinar archivos .npy dentro de un directorio en un solo archivo.

    Parámetros:
    - data_dir (str): Ruta del directorio donde se encuentran los archivos de características.
    - output_filename (str): Nombre del archivo de salida que contendrá todas las características combinadas.

    Justificación:
    - Los archivos de características extraídas están separados en múltiples archivos .npy.
    - Para aplicar reducción de dimensionalidad con PCA, es más eficiente trabajar con un solo archivo consolidado.
    - Se necesita garantizar que todos los archivos combinados tengan la misma cantidad de columnas para evitar errores de concatenación.
    """

    # Obtener lista de archivos .npy en el directorio
    npy_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
    features_list = []

    for file in npy_files:
        file_path = os.path.join(data_dir, file)
        array = np.load(file_path, allow_pickle=True)  # Cargar con allow_pickle por seguridad

        # Verificar si el archivo está vacío o no tiene la estructura esperada
        if array is None or array.size == 0:
            print(f"Archivo vacío ignorado: {file}")
            continue

        if len(array.shape) < 2:
            print(f"Archivo ignorado por ser de dimensión incorrecta: {file} - {array.shape}")
            continue  # Ignorar archivos que no sean bidimensionales

        if array.shape[1] != 512:
            print(f"Archivo ignorado por tamaño incorrecto: {file} - {array.shape}")
            continue  # Ignorar archivos con diferentes columnas

        features_list.append(array)

    # Verificar si hay datos válidos para evitar guardar archivos vacíos
    if not features_list:
        print(f"No hay datos válidos en {data_dir}, no se creará el archivo {output_filename}.")
        return

    # Concatenar todos los archivos filtrados en una única matriz
    features_combined = np.vstack(features_list)

    # Guardar el archivo combinado
    np.save(os.path.join(data_dir, output_filename), features_combined)
    print(f"Archivo {output_filename} creado en {data_dir}")

# Ejecutar la función para los conjuntos de entrenamiento (train) y validación (val)
combine_features_v2("data/r2plus1d_18_16_kinetics_train", "features_train.npy")
combine_features_v2("data/r2plus1d_18_16_kinetics_val", "features_val.npy")
