import numpy as np
import os

def check_shapes(data_dir):
    npy_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
    for file in npy_files:
        file_path = os.path.join(data_dir, file)
        try:
            array = np.load(file_path)
            print(f"{file}: {array.shape}")
        except Exception as e:
            print(f"❌ Error al cargar {file}: {e}")

# Verificar train y val
print("📂 **Verificando archivos en train**")
check_shapes("data/r2plus1d_18_16_kinetics_train")

print("\n📂 **Verificando archivos en val**")
check_shapes("data/r2plus1d_18_16_kinetics_val")
