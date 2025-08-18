# DDG Network Embedding Extraction

This directory contains scripts to extract embeddings from the DDG RDE Network model for protein-protein interaction analysis.

## Problem Solved

The original script had a critical bug where it was using loop indices (0, 1, 2, ...) instead of the actual complex IDs from the ground truth txt files. This caused mismatches like:
- Expected: `692_1N8O` (from txt files)
- Actual: `675_1N8O` (wrong index)

## Fixed Scripts

### 1. `extract_ddg_network_embeddings.py` (MAIN SCRIPT)
The main extraction script with the following improvements:

- **Exact Complex ID Matching**: Loads complex IDs from txt files and validates against dataset output
- **PDB File Verification**: Ensures all required PDB files exist before processing
- **Proper Fold Indexing**: Correctly maps fold indices (0-9) to directory names (fold_1, fold_2, etc.)
- **Split Mapping**: Correctly maps 'val' split to 'test' txt files
- **Validation**: Tracks processed complexes and reports any missing ones
- **Debugging**: Shows dataset structure and available fields

### 2. `test_complex_id_loading.py` (TEST SCRIPT)
Run this **before** the main extraction to verify:
- Complex ID loading from txt files works
- PDB file verification works
- Directory structure is correct

### 3. `validate_extracted_embeddings.py` (VALIDATION SCRIPT)
Run this **after** the main extraction to verify:
- All expected files were created
- File names match ground truth exactly
- No missing embeddings

## Usage

### Step 1: Test Setup
```bash
python test_complex_id_loading.py
```

### Step 2: Extract Embeddings
```bash
python extract_ddg_network_embeddings.py \
    --ckpt /path/to/checkpoint.pt \
    --config /path/to/config.yaml \
    --output_dir /path/to/output \
    --device cuda
```

### Step 3: Validate Results
```bash
python validate_extracted_embeddings.py \
    --output_dir /path/to/output \
    --folds_dir cross_validation_folds_final/60_percent \
    --threshold 60
```

## Expected Output Structure

For each fold and split, you should get:
```
fold_1/
├── train/
│   ├── 10_1ACB_wt.npy
│   ├── 10_1ACB_mt.npy
│   ├── 10_1ACB_wt_resmap.pkl
│   ├── 10_1ACB_mt_resmap.pkl
│   ├── 11_1ACB_wt.npy
│   └── ...
└── test/
    ├── 692_1N8O_wt.npy
    ├── 692_1N8O_mt.npy
    ├── 692_1N8O_wt_resmap.pkl
    ├── 692_1N8O_mt_resmap.pkl
    └── ...
```

## Key Fixes Applied

1. **Complex ID Source**: Now reads from txt files instead of using loop indices
2. **Fold Indexing**: Fixed 0-9 → 1-10 mapping for directory names
3. **Split Mapping**: Fixed 'val' → 'test' mapping for txt files
4. **Validation**: Added comprehensive validation against ground truth
5. **Error Handling**: Better error reporting and debugging information

## Verification

The scripts ensure that:
- Every complex ID in the txt files has exactly 4 corresponding files
- File names match the txt files exactly (e.g., `692_1N8O`, not `675_1N8O`)
- All PDB files exist before processing
- No complexes are missed or duplicated

## Troubleshooting

If you encounter issues:

1. **Run the test script first** to verify setup
2. **Check the debug output** to see dataset structure
3. **Verify PDB files exist** in the wildtype directory
4. **Run validation script** to identify any missing files
5. **Check fold directory structure** matches expected naming

## Important Notes

- The complex IDs (like `692_1N8O`) are **UNIVERSAL** and must match exactly across all files
- Each complex requires exactly 4 files (2 .npy + 2 .pkl)
- The residue mapping files (.pkl) ensure exact residue correspondence
- The script validates against ground truth txt files to prevent mismatches 