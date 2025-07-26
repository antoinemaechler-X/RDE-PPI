import numpy as np
import pickle

data = np.load("ddg_network_embeddings_matching/fold_4/test/2097_1DAN_HL_UT_YU67A_wt.npy")

print(data.shape)
print(data)

# --- Paste your .pkl file path below to read and print the residue mapping ---
pkl_path = "ddg_network_embeddings_matching/fold_4/test/2097_1DAN_HL_UT_YU67A_wt.pkl"  # <-- Paste your .pkl file path here, e.g. "ddg_network_embeddings/fold_0/test/0_1REW_AB_C_DA19A,DB19A_resmap.pkl"
if pkl_path:
    with open(pkl_path, 'rb') as f:
        residue_map = pickle.load(f)
    print("Residue mapping:")
    print(residue_map)

