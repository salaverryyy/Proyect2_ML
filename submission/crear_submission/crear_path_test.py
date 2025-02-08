import pandas as pd

# Cargar test_subset.csv para obtener los IDs
test_csv = "data/etiquetas_reales/test_subset.csv"
df = pd.read_csv(test_csv)

# Verificar que la columna correcta exista
if "youtube_id" not in df.columns:
    raise ValueError(f"Error: 'youtube_id' no encontrado en {test_csv}. Revisa los nombres de las columnas.")

# Guardar los IDs en path_test.txt
path_test_file = "data/r2plus1d_18_16_kinetics/path_test.txt"
df["youtube_id"].to_csv(path_test_file, index=False, header=False)

print(f"✅ Archivo creado: {path_test_file} con {len(df)} IDs.")
