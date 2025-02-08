import pandas as pd

# Rutas de archivos
path_test = "data/r2plus1d_18_16_kinetics/path_test.txt"
test_csv = "data/etiquetas_reales/test_subset.csv"

# Contar líneas en path_test.txt
with open(path_test, "r") as f:
    test_ids = f.readlines()
print(f"Total de IDs en path_test.txt: {len(test_ids)}")

# Contar IDs en test_subset.csv
df = pd.read_csv(test_csv)
print(f"Total de IDs en test_subset.csv: {df.shape[0]}")
