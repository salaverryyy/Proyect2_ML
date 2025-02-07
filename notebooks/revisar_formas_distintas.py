import numpy as np
import os
from collections import defaultdict

def check_shapes_v2(data_dir):
    npy_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
    shape_counts = defaultdict(int)

    for file in npy_files:
        file_path = os.path.join(data_dir, file)
        try:
            array = np.load(file_path)
            shape_counts[array.shape] += 1
        except Exception as e:
            print(f" Error al cargar {file}: {e}")

    print(f"\n**Formas de archivos en {data_dir}:**")
    for shape, count in shape_counts.items():
        print(f"{shape}: {count} archivos")

# Verificar train y val
check_shapes_v2("data/r2plus1d_18_16_kinetics_train")
check_shapes_v2("data/r2plus1d_18_16_kinetics_val")
