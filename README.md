# DESPOT

Direction-Enhanced Scoring POTentials: anisotropic knowledge-based potential for scoring protein-ligand interactions.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/robin-poelmans/DESPOT.git
cd DESPOT
```

### 2. Create Conda Environment

Install the required dependencies using the provided environment file:
```bash
conda env create -f environment.yml
conda activate DESPOT
```

### 3. Install DESPOT Package

Install DESPOT in development mode:
```bash
pip install -e .
```

### 4. Download Models and Datasets

Download the pre-trained models and training/test datasets:
```bash
bash download_data.sh
```

This script will download:
- Pre-trained DESPOT models
- CROWN training dataset
- CASF benchmark test dataset

## Usage

### Inference

To score a protein-ligand complex:
```bash
python scripts/score_complex.py -p 1e66_protein.mol2 -l 1e66_decoys.mol2 -o test.csv -m full --bfac
```

**Options:**

- `-p` / `--protein`: Path (relative or full) to receptor .mol2 file **[REQUIRED]**
- `-l` / `--ligand`: Path (relative or full) to ligand (multi-)mol2 file **[REQUIRED]**
- `-o` / `--outpath`: Path (relative or full) to output CSV file **[REQUIRED]**
- `-m` / `--mode`: Which DESPOT mode to use for inference. Choose between `{full, iso, ds}`. Default: `full` **[OPTIONAL]**
- `--bfac`: Set this flag to create a subdirectory that stores separate PDB files for each ligand pose, with atom-wise scores stored as B-factors **[OPTIONAL]**

**Example Output (`test.csv`):**
```csv
ligand,score
1e66_404,-143.505772
1e66_505,-43.276867
1e66_862,-62.781874
1e66_247,-177.097640
1e66_591,38.887140
1e66_661,-41.130724
```

The output CSV contains ligand identifiers (parsed from the mol2 file) and their corresponding binding scores. Lower (more negative) scores indicate stronger predicted binding affinity.

### Training

To retrain DESPOT from scratch, follow these steps:

#### 1. Preprocess Training Data

Preprocess the CROWN training dataset:
```bash
python scripts/preprocess_crown.py
```

#### 2. Train Models

Train new DESPOT models:
```bash
python scripts/train_despot.py
```

#### 3. Benchmark and Evaluate

Evaluate trained models on the CASF benchmark set and generate performance plots:
```bash
python scripts/benchmark_casf.py
```

## Citation

If you use DESPOT in your research, please cite:
```
(Add citation here)
```

## License

This dataset is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Contact

For questions or issues, contact:

Robin Poelmans
Laboratory for Biomolecular Modelling and Design, Department of Chemistry, KU Leuven
robin.poelmans@kuleuven.be
