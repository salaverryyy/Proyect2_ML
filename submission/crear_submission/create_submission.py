import numpy as np
import pandas as pd

# Cargar los IDs de test desde test_subset_10.csv
test_ids_file = "data/etiquetas_Reales/test_subset_10.csv"
df_test = pd.read_csv(test_ids_file)

# Extraer los 805 IDs directamente
youtube_ids = df_test["youtube_id"].tolist()

# Cargar etiquetas de clustering
test_labels_file = "data/dbscan_clustering/test_tsne_2d_dbscan_labels.npy"

labels = np.load(test_labels_file)

# Recortar etiquetas a 805 (por si acaso)
labels = labels[:805]

# Crear DataFrame de submission
submission = pd.DataFrame({"youtube_id": youtube_ids, "label": labels})

# Guardar CSV
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(f" Archivo de submission guardado con {len(submission)} filas. VAMOOOOS")
