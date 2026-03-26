# DESPOT

Direction-Enhanced Scoring POTentials: anisotropic knowledge-based potential for scoring protein-ligand interactions.

## About

Knowledge-based potentials (KBPs) are widely used to score protein–ligand interactions, but existing methods are isotropic — they capture only distance dependencies and ignore the directional preferences that govern molecular recognition. DESPOT instead naturally supports both directional scoring and steric exclusion, making it a tool that can be used for both post-scoring of docking results, as well as ligand-independent generation of molecular interaction fields. On the CASF-2016 benchmark, DESPOT substantially outperforms isotropic KBPs across all pose-discrimination and virtual screening tasks.

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

Download the pre-trained DESPOT models and test datasets:
```bash
bash download_data.sh
```

This script will download:
- Pre-trained DESPOT models
- CASF benchmark test set

## Usage

### Inference

#### 1. Pose scoring

To score a protein-ligand complex:
```bash
python scripts/score_complex.py -p 6hdt_receptor.mol2 -l 6hdt_ligand.mol2 -o test_score.csv -m full --database CROWN_leaky --bfac
```

**Options:**

- `-p` / `--protein`: Path (relative or full) to receptor .mol2 file **[REQUIRED]**
- `-l` / `--ligand`: Path (relative or full) to ligand (multi-)mol2 file **[REQUIRED]**
- `-o` / `--outpath`: Path (relative or full) to output CSV file **[REQUIRED]**
- `-m` / `--mode`: Which DESPOT mode to use for inference. Choose between `{full, ds}`. Default: `full` **[OPTIONAL]**
- `--bfac`: Set this flag to create a subdirectory that stores separate PDB files for each ligand pose, with atom-wise scores stored as B-factors **[OPTIONAL]**
- `--database`: Which training source to use for inference. Choose between [CROWN_train, CROWN_Xtal, CROWN_leaky]. Default: CROWN_train **[OPTIONAL]**

**Example Output (`test_score.csv`):**
```csv
ligand,score
1.E,-428.293167
```

The output CSV contains ligand identifiers (parsed from the mol2 file) and their corresponding binding scores. Lower (more negative) scores indicate stronger predicted binding affinity.

#### 2. MIF generation

To create a MIF channel map:
```bash
python scripts/make_voxel_channel.py --protein 6hdt_receptor.mol2 --pocket 6hdt_pocket.pqr --channel O.co2_1 --output test_voxel.pdb --database CROWN_leaky
```

**Options:**

- `--protein`: Path (relative or full) to receptor .mol2 file **[REQUIRED]**
- `--pocket`: Path (relative or full) to FPocket .pqr file (used for pocket definition) **[REQUIRED]**
- `--channel`: Name of ligand atom type you want the MIF channel from (e.g., O.co2_1). See supplementary information of paper for full atom type definitions. **[REQUIRED]**
- `--output`: Path (relative or full) to output .pdb file **[REQUIRED]**
- `--database`: Which training source to use for inference. Choose between [CROWN_train, CROWN_Xtal, CROWN_leaky]. Default: CROWN_train **[OPTIONAL]**

#### 3. Plotting interaction types
A Jupyter notebook is provided for plotting pairwise interactions.
The function *plot_anisotropy* takes 3 arguments:

- `p_types_list`: L protein atom types. See supplementary information of paper for full atom type definitions.
- `l_types_list`: L ligand atom types. See supplementary information of paper for full atom type definitions.
- `distance_list`: L distances used for creating Mollweide projections.

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
python scripts/count_interactions.py --database CROWN_train  # Options are: [CROWN_train, CROWN_Xtal, CROWN_leaky]
python scripts/train_despot.py --database CROWN_train  # Options are: [CROWN_train, CROWN_Xtal, CROWN_leaky]
```

#### 3. Benchmarking

Evaluate trained models on the CASF benchmark set and generate performance plots:
```bash
python scripts/benchmark_casf.py --database CROWN_train  # Options are: [CROWN_train, CROWN_Xtal, CROWN_leaky]
```

## CROWN Dataset

The CROWN dataset used for training and evaluation is maintained separately.
More information, updates, and access instructions are available on the official website:

## Citation

If you use DESPOT in your research, please cite:
```
(Add citation here)
```

## License

### DESPOT

This project is licensed under the MIT License.

### CROWN Dataset

The CROWN dataset is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Contact

For questions or issues, contact:

Robin Poelmans
Laboratory for Biomolecular Modelling and Design, Department of Chemistry, KU Leuven
robin.poelmans@kuleuven.be
