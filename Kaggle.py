import os
import shutil
from pathlib import Path
import pandas as pd

train_p = "C:\\Users\\Matías\\Videos\\p2\\project-3-clustering-2025-0\\train_subset_10.csv"
val_p = "C:\\Users\\Matías\\Videos\\p2\\project-3-clustering-2025-0\\val_subset_10.csv"
test_p = "C:\\Users\\Matías\\Videos\\p2\\project-3-clustering-2025-0\\test_subset_10.csv"

train_df = pd.read_csv(train_p)
val_df = pd.read_csv(val_p)
test_df = pd.read_csv(test_p)

print(train_df.head())
print(val_df.head())
print(test_df.head())

#Como necesitamos cada nombre en cada linea como string incluido el .mp4 necesitamos renombrar y escribir los paths de forma de [,,,] en 3 text (para cada stage)
def rename(directories: list[str]): 

    print("Iniciando el proceso de renombrado...\n")

    for directory in directories:
        dir_path = Path(directory)

        if not dir_path.exists():
            print(f"El directorio {directory} no existe. Saltando...")
            continue

        print(f"Renombrando archivos en {directory}...")

        for file_path in dir_path.glob("*.mp4"):
            filename = file_path.name
            if "_" in filename:
                video_id = filename.split('_')[0]
                new_filename = f"{video_id}.mp4"
                new_file_path = dir_path / new_filename

                if file_path != new_file_path and not new_file_path.exists():
                    shutil.move(file_path, new_file_path) 
                    print(f"Renombrado: {file_path} → {new_file_path}")

        print(f"Renombrado completado en {directory}.\n")

def video_paths_txt(df: pd.DataFrame, directory_path: str, name: str):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a Pandas DataFrame")

    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"El directorio {directory_path} no existe.")

    # Generar rutas con formato correcto (una por línea, sin `[` ni `,`)
    paths = [str(directory / f"{video_id}.mp4").replace("\\", "/") for video_id in df['youtube_id']]
    
    output_file = Path(f"./path_{name}.txt")
    output_file.write_text("\n".join(paths), encoding="utf-8")

    print(f"Archivo generado correctamente: {output_file}")

directories = [r'C:\Users\videos\train_subset', 
               r'C:\Users\videos\val_subset', 
               r'C:\Users\videos\test_subset']

train_directory = r"C:\Users\videos\train_subset"
val_directory = r"C:\Users\videos\val_subset"
test_directory = r"C:\Users\videos\test_subset"


rename(directories)

video_paths_txt(val_df, val_directory, 'val')
video_paths_txt(train_df, train_directory, 'train')
video_paths_txt(test_df, test_directory, 'test')

#Ahora ya tenemos los 3 archivos txt hacemos lo siguiente

# # clone the repo and change the working directory
# git clone https://github.com/v-iashin/video_features.git
# cd video_features

# conda create -n video_features
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# conda install -c conda-forge omegaconf scipy tqdm pytest opencv
# conda install -c conda-forge av



# # extract r(2+1)d features for the sample videos
# python main.py \
#     feature_type=r21d \
#     device="cuda:0" \
#     video_paths="[./sample/v_ZNVhz7ctTq0.mp4, ./sample/v_GGSY1Qvo990.mp4]"
# # if you have many GPUs, just run this command from another terminal with another device
# # device can also be "cpu"

#video_paths seran nuestros txt, tambien se tuvo que instalar conda (https://www.anaconda.com/download/success)

