# DESPOT

**Direction-Enhanced Scoring POTentials**

DESPOT is an anisotropic knowledge-based potential for scoring protein–ligand interactions. Unlike conventional isotropic approaches that capture only distance dependencies, DESPOT incorporates the directional preferences that govern molecular recognition. This enables both post-scoring of docking results and ligand-independent generation of molecular interaction fields (MIFs). On the CASF-2016 benchmark, DESPOT substantially outperforms isotropic knowledge-based potentials across all pose-discrimination and virtual screening tasks.

## Installation

### 1. Clone and set up the environment

```bash
git clone https://github.com/robin-poelmans/DESPOT.git
cd DESPOT
conda env create -f environment.yml
conda activate DESPOT
pip install -e .
```

### 2. Download models and datasets

Download the pre-trained DESPOT models and CASF benchmark test set:

```bash
bash download_dataset.sh
```

## Usage

### Inference

#### Pose scoring

Score a protein–ligand complex:

```bash
python scripts/score_complex.py \
  -p 6hdt_receptor.mol2 \
  -l 6hdt_ligand.mol2 \
  -o test_score.csv \
  -m full \
  --database CROWN_leaky \
  --bfac
```

| Option | Description |
|--------|-------------|
| `-p` / `--protein` | Path to receptor `.mol2` file **(required)** |
| `-l` / `--ligand` | Path to ligand (multi-)`.mol2` file **(required)** |
| `-o` / `--outpath` | Path to output CSV file **(required)** |
| `-m` / `--mode` | DESPOT mode: `full` or `ds` (default: `full`) |
| `--database` | Training source: `CROWN_train`, `CROWN_Xtal`, or `CROWN_leaky` (default: `CROWN_train`) |
| `--bfac` | Write per-pose PDB files with atom-wise scores as B-factors |

The output CSV contains ligand identifiers and binding scores, where lower (more negative) scores indicate stronger predicted binding affinity.

#### MIF generation

Generate a molecular interaction field channel map:

```bash
python scripts/make_voxel_channel.py \
  --protein 6hdt_receptor.mol2 \
  --pocket 6hdt_pocket.pqr \
  --channel O.co2_1 \
  --output test_voxel.pdb \
  --database CROWN_leaky
```

| Option | Description |
|--------|-------------|
| `--protein` | Path to receptor `.mol2` file **(required)** |
| `--pocket` | Path to FPocket `.pqr` file for pocket definition **(required)** |
| `--channel` | Ligand atom type for the MIF channel (e.g., `O.co2_1`) **(required)** |
| `--output` | Path to output `.pdb` file **(required)** |
| `--database` | Training source: `CROWN_train`, `CROWN_Xtal`, or `CROWN_leaky` (default: `CROWN_train`) |

See the supplementary information of the paper for full atom type definitions.

#### Plotting interaction types

A Jupyter notebook is provided for plotting pairwise interactions. The `plot_anisotropy` function takes three arguments: a list of protein atom types, a list of ligand atom types, and a list of distances for creating Mollweide projections.

### Training

To retrain DESPOT from scratch:

```bash
# 1. Count interactions and train models
python scripts/count_interactions.py --database CROWN_train
python scripts/train_despot.py --database CROWN_train

# 2. Benchmark on CASF
python scripts/benchmark_casf.py --database CROWN_train
```

All training scripts accept `--database` with options `CROWN_train`, `CROWN_Xtal`, or `CROWN_leaky`.

## CROWN Dataset

DESPOT is trained on the [CROWN](https://github.com/KUL-LBMD/CROWN) (Curated Repository Of Well-resolved Non-covalent interactions) dataset, a curated collection of 153,005 protein–ligand complexes. The dataset can be browsed and searched interactively at [crown.lbmd.be](https://crown.lbmd.be), or downloaded in bulk from [Zenodo](https://zenodo.org/records/19334311).

## Citation

If you use DESPOT in your work, please cite:

DESPOT: Direction-Enhanced Scoring POTentials
Robin Poelmans, Bence Bruncsics, Adam Arany, Wout Van Eynde, Ahmed Shemy, Yves Moreau, Arnout RD Voet
bioRxiv 2026.03.31.714140; doi: [https://doi.org/10.64898/2026.03.31.714140](https://doi.org/10.64898/2026.03.31.714140)

## License

DESPOT is licensed under the [MIT License](LICENSE). The CROWN dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Contact

Robin Poelmans
Laboratory for Biomolecular Modelling and Design, Department of Chemistry, KU Leuven
[robin.poelmans@kuleuven.be](mailto:robin.poelmans@kuleuven.be)
