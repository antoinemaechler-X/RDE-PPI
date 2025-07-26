import os
import shutil
import pandas as pd
import re

# Paths
input_dir = "ddg_network_embeddings"
output_dir = "ddg_network_embeddings_matching"
skempi_csv = "data/SKEMPI_v2/skempi_v2.csv"

# Read SKEMPI2.csv and build mapping: (complex, mutation) -> index
skempi_df = pd.read_csv(skempi_csv, sep=';')
skempi_map = {}
for idx, row in skempi_df.iterrows():
    # Use #Pdb and Mutation(s)_cleaned as the key
    key = (str(row['#Pdb']), str(row['Mutation(s)_cleaned']))
    skempi_map[key] = idx

# Helper to extract complex and mutation from filename
# Example: 0_1REW_AB_C_DA19A,DB19A_mt.npy -> ('1REW_AB_C', 'DA19A,DB19A')
def extract_complex_mut(filename):
    # Remove fold/train/test and extension
    base = os.path.basename(filename)
    # Remove _wt/_mt and extension
    base = re.sub(r'_(wt|mt)(\.npy|_resmap\.pkl)$', '', base)
    # Split by first underscore (idx) and last underscore (mutstr)
    parts = base.split('_')
    if len(parts) < 3:
        return None, None
    # idx, complex, mutstr (mutstr may contain underscores if multiple mutations)
    idx = parts[0]
    complex_name = '_'.join(parts[1:-1])
    mutstr = parts[-1]
    return complex_name, mutstr

files_processed = 0
files_skipped = 0
errors = 0

# Walk through all files in input_dir
for fold in range(10):
    for split in ['train', 'test']:
        in_dir = os.path.join(input_dir, f"fold_{fold}", split)
        out_dir = os.path.join(output_dir, f"fold_{fold}", split)
        os.makedirs(out_dir, exist_ok=True)
        if not os.path.exists(in_dir):
            print(f"Input directory does not exist: {in_dir}")
            continue
        for fname in os.listdir(in_dir):
            if not (fname.endswith('.npy') or fname.endswith('.pkl')):
                continue
            # Extract complex and mutation
            complex_name, mutstr = extract_complex_mut(fname)
            if complex_name is None or mutstr is None:
                print(f"[SKIP] Could not parse complex/mutstr from: {fname}")
                files_skipped += 1
                continue
            # Try to find in SKEMPI map
            key = (complex_name, mutstr)
            if key not in skempi_map:
                print(f"[SKIP] No match in SKEMPI for {key} (from {fname})")
                files_skipped += 1
                continue
            idx = skempi_map[key]
            # Determine wt/mt and extension
            if '_wt' in fname:
                suffix = '_wt'
            elif '_mt' in fname:
                suffix = '_mt'
            else:
                suffix = ''
            ext = '.npy' if fname.endswith('.npy') else '.pkl'
            # Compose new filename
            new_fname = f"{idx}_{complex_name}_{mutstr}{suffix}{ext}"
            # Copy file
            try:
                shutil.copy2(os.path.join(in_dir, fname), os.path.join(out_dir, new_fname))
                files_processed += 1
            except Exception as e:
                print(f"[ERROR] Failed to copy {fname} to {new_fname}: {e}")
                errors += 1

print(f"\nSummary:")
print(f"  Files processed: {files_processed}")
print(f"  Files skipped:   {files_skipped}")
print(f"  Errors:          {errors}") 