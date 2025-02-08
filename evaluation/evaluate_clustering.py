import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
import os

# Directorios
clustering_dirs = {
    "dbscan": "data/dbscan_clustering",
    "kmeans": "data/kmeans_clustering"
}
features_dir = "data/reduction"
labels_dir = "etiquetas_reales"
video_id_files = {  # Archivos que contienen los IDs usados en clustering
    "train": "data/r2plus1d_18_16_kinetics_train/path_train.txt",
    "val": "data/r2plus1d_18_16_kinetics_val/path_val.txt"
}

# Cargar etiquetas reales desde CSV
def load_real_labels():
    label_files = {
        "train": os.path.join(labels_dir, "train_subset.csv"),
        "val": os.path.join(labels_dir, "val_subset.csv"),
    }
    
    real_labels = {}
    for key, path in label_files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "youtube_id" in df.columns and "label" in df.columns:
                real_labels[key] = df.set_index("youtube_id")["label"].to_dict()
            else:
                print(f"Advertencia: {path} no contiene las columnas necesarias.")
        else:
            print(f"Advertencia: Archivo {path} no encontrado.")

    return real_labels

# Cargar IDs usados en clustering
def load_video_ids():
    video_ids = {}
    for key, path in video_id_files.items():
        if os.path.exists(path):
            with open(path, "r") as f:
                video_ids[key] = [line.strip() for line in f.readlines()]
        else:
            print(f"Advertencia: Archivo {path} no encontrado.")
    return video_ids

# Cargar etiquetas reales e IDs usados
real_labels = load_real_labels()
video_ids_used = load_video_ids()

# Evaluación de clustering
for method, path in clustering_dirs.items():
    print(f"\nEvaluando {method.upper()}...\n")
    
    for file in sorted(os.listdir(path)):
        if not file.endswith("_labels.npy"):
            continue
        
        feature_file = file.replace(f"_{method}_labels.npy", ".npy")  
        feature_path = os.path.join(features_dir, feature_file)
        
        if not os.path.exists(feature_path):
            print(f"{file}: ERROR - Archivo de features no encontrado.")
            continue
        
        labels = np.load(os.path.join(path, file))
        features = np.load(feature_path)

        dataset_type = "train" if "train" in file else "val" if "val" in file else None

        if dataset_type and dataset_type in real_labels and dataset_type in video_ids_used:
            # Filtrar etiquetas reales para que coincidan con las muestras usadas en clustering
            filtered_labels = [real_labels[dataset_type].get(yid, -1) for yid in video_ids_used[dataset_type]]
            ground_truth = np.array([gt for gt in filtered_labels if gt != -1])  # Eliminar valores no encontrados
            
            # Asegurar que ground_truth y labels tengan el mismo tamaño
            if len(ground_truth) == len(labels) and len(set(ground_truth)) > 1 and len(set(labels)) > 1:
                silhouette = silhouette_score(features, labels)
                rand_index = adjusted_rand_score(ground_truth, labels)
                mutual_info = adjusted_mutual_info_score(ground_truth, labels)
                
                print(f"{file}: {silhouette:.4f} | {rand_index:.4f} | {mutual_info:.4f}")
            else:
                print(f"{file}: No suficientes etiquetas válidas para calcular métricas.")
        else:
            if len(set(labels)) > 1:
                silhouette = silhouette_score(features, labels)
                print(f"{file}: {silhouette:.4f}")
            else:
                print(f"{file}: Un solo cluster, no se puede calcular Silhouette Score.")
