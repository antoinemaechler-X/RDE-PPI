import numpy as np
import pickle

data = np.load("ddg_network_embeddings_wildtype_only/fold_1/train/104_1BRS_mt.npy")

print(data.shape)
print(data)

# --- Paste your .pkl file path below to read and print the residue mapping ---
pkl_path = "ddg_network_embeddings_wildtype_only/fold_1/train/104_1BRS_mt_resmap.pkl"
if pkl_path:
    with open(pkl_path, 'rb') as f:
        residue_map = pickle.load(f)
    print("Residue mapping:")
    print(residue_map)

