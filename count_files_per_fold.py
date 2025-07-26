import os

output_dir = "ddg_network_embeddings"  # Change if your output directory is different
num_folds = 10
splits = ["train", "test"]

for fold in range(num_folds):
    print(f"Fold {fold}:")
    for split in splits:
        split_dir = os.path.join(output_dir, f"fold_{fold}", split)
        if not os.path.exists(split_dir):
            print(f"  {split}: directory does not exist")
            continue
        # Count only .npy files (not .pkl)
        npy_files = [f for f in os.listdir(split_dir) if f.endswith('.npy')]
        print(f"  {split}: {len(npy_files)} .npy files")
    print() 