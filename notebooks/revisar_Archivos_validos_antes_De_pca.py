import numpy as np

train_data = np.load("data/r2plus1d_18_16_kinetics_train/features_train.npy", allow_pickle=True)
val_data = np.load("data/r2plus1d_18_16_kinetics_val/features_val.npy", allow_pickle=True)
test_data = np.load("data/r2plus1d_18_16_kinetics/features_test.npy", allow_pickle=True)

print("Train shape:", train_data.shape)
print("Validation shape:", val_data.shape)
print("Test shape:", test_data.shape)
