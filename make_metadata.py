from src.config import DATA_DIR, SOURCE_DB_PATH

import pandas as pd
import os
import re
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

def safe_murcko(smi):
	try:
		scaffold = MurckoScaffoldSmiles(smi)
		return scaffold if scaffold else None
	except:
		return None

df = pd.read_parquet(SOURCE_DB_PATH / 'index' / 'annotation_table.parquet')
print(len(df))
df['basename'] = df['system_id'].str.lower() + '_' + df['ligand_instance_chain'].str.lower()
df = df.drop_duplicates(subset = ['entry_pdb_id', 'ligand_unique_ccd_code'])

# CROWN
crown_list = os.listdir(DATA_DIR / 'CROWN' / 'processed_systems')
crown_df = df[df['basename'].isin(crown_list)]
crown_df = crown_df.drop_duplicates(subset = ['entry_pdb_id', 'ligand_unique_ccd_code'])

# PDB to taxonomy
tax_df = pd.read_csv('/media/drives/drive3/robin/EmbContext/mapping/pdb_chain_taxonomy.tsv', sep = '\t')
tax_df = tax_df.drop_duplicates(subset = ['PDB'], keep = 'first')
df = df.merge(tax_df, left_on = ['entry_pdb_id'], right_on = ['PDB'], how = 'inner')
crown_df = crown_df.merge(tax_df, left_on = ['entry_pdb_id'], right_on = ['PDB'], how = 'inner')

# Bemis-Murcko mapping
df['murcko_scaffold'] = df['ligand_rdkit_canonical_smiles'].apply(safe_murcko)
crown_df['murcko_scaffold'] = crown_df['ligand_rdkit_canonical_smiles'].apply(safe_murcko)

columns_to_select = ['basename', 'entry_pdb_id', 'system_id', 'ligand_instance_chain', 'entry_resolution', 'system_ligand_validation_average_rsr', 'system_ligand_validation_average_rscc',
	'system_pocket_UniProt', 'system_pocket_CATH', 'ligand_unique_ccd_code', 'ligand_rdkit_canonical_smiles', 'entry_determination_method', 'ligand_num_unresolved_heavy_atoms', 'ligand_is_covalent', 'ligand_is_artifact', 'ligand_is_ion',
	'TAX_ID', 'ligand_num_rot_bonds', 'ligand_num_hbd', 'ligand_num_hba', 'ligand_num_heavy_atoms', 'murcko_scaffold']

subset = df[columns_to_select]
crown_subset = crown_df[columns_to_select]

subset.to_csv(DATA_DIR / 'CROWN' / 'metadata' / 'plinder_full.csv', index = False, float_format = '%.3f')
crown_subset.to_csv(DATA_DIR / 'CROWN' / 'metadata' / 'crown_full.csv', index = False, float_format = '%.3f')


# HiQBind
hiqbind_df = pd.read_csv('/media/drives/drive3/robin/HiQBind/figshare/hiqbind_metadata.csv')
merged = subset.merge(
    hiqbind_df[['PDBID', 'Ligand Name']].drop_duplicates(),
    left_on=['entry_pdb_id', 'ligand_unique_ccd_code'],
    right_on=['PDBID', 'Ligand Name'],
    how='inner'
)

# Keep the row with best RSR per (PDBID, Ligand Name)
hiqbind_subset = (
    merged
    .sort_values('system_ligand_validation_average_rsr', ascending=True)
    .drop_duplicates(subset=['PDBID', 'Ligand Name'], keep='first')
    .drop(columns=['PDBID', 'Ligand Name'])
    .reset_index(drop=True)
)

hiqbind_subset = hiqbind_subset[columns_to_select]
hiqbind_subset.to_csv(DATA_DIR / 'CROWN' / 'metadata' / 'hiqbind.csv', index = False, float_format = '%.3f')

# PDBBind
# Parse the PDBbind index file
rows = []
with open('/media/drives/drive3/robin/index/INDEX_general_PL_data.2020') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('='):
            continue
        pdb_code = line[:4].strip()
        # Extract the string in parentheses, e.g. (BAN), (2-mer), (5-mer)
        match = re.search(r'\(([^)]+)\)', line)
        ligand_name = match.group(1).strip() if match else None
        rows.append({'pdb_id': pdb_code, 'ligand_name': ligand_name})

pdbbind_df = pd.DataFrame(rows).drop_duplicates()

# Determine if ligand_name is a valid 3-letter CCD code (3 uppercase alphanumeric chars)
pdbbind_df['is_ccd'] = pdbbind_df['ligand_name'].str.match(r'^[A-Z0-9]{3}$', na=False)
pdbbind_df.to_csv('pdbbind_full.csv')

# --- Entries with valid 3-letter CCD codes: match on PDB + ligand ---
ccd_entries = pdbbind_df[pdbbind_df['is_ccd']]
merged_ccd = subset.merge(
    ccd_entries[['pdb_id', 'ligand_name']].drop_duplicates(),
    left_on=['entry_pdb_id', 'ligand_unique_ccd_code'],
    right_on=['pdb_id', 'ligand_name'],
    how='inner'
)

best_ccd = (
    merged_ccd
    .sort_values('system_ligand_validation_average_rsr', ascending=True)
    .drop_duplicates(subset=['pdb_id', 'ligand_name'], keep='first')
    .drop(columns=['ligand_name'])
)

# --- Entries without valid CCD codes: match on PDB only, take best RSR ---
non_ccd_entries = pdbbind_df[~pdbbind_df['is_ccd']]
merged_non_ccd = subset.merge(
    non_ccd_entries[['pdb_id']].drop_duplicates(),
    left_on=['entry_pdb_id'], right_on = ['pdb_id'],
    how='inner'
)
best_non_ccd = (
    merged_non_ccd
    .sort_values('ligand_num_heavy_atoms', ascending=False)
    .drop_duplicates(subset=['pdb_id'], keep='first')
)

# Combine both subsets
pdbbind_subset = (
    pd.concat([best_ccd, best_non_ccd], ignore_index=True)
    .drop_duplicates(subset=['basename'])
    .drop(columns=['pdb_id'])
    .reset_index(drop=True)
)

pdbbind_subset = pdbbind_subset[columns_to_select]
pdbbind_subset.to_csv(DATA_DIR / 'CROWN' / 'metadata' / 'pdbbind.csv', index = False, float_format = '%.3f')

### BioLiP2 ###
# Parse index file
rows = []
with open('/media/drives/drive3/robin/biolip/BioLiP_nr.txt') as f:
    for line in f:
        line = line.strip()
        parts = line.split()
        pdb_code = parts[0]
        ligand_name = parts[4]
        rows.append({'pdb_id': pdb_code, 'ligand_name': ligand_name})

biolip_df = pd.DataFrame(rows).drop_duplicates()
biolip_df['is_ccd'] = biolip_df['ligand_name'].str.match(r'^[A-Z0-9]{2,3}$', na=False)

# --- Entries with valid 3-letter CCD codes: match on PDB + ligand ---
ccd_entries = biolip_df[biolip_df['is_ccd']]
merged_ccd = subset.merge(
    ccd_entries[['pdb_id', 'ligand_name']].drop_duplicates(),
    left_on=['entry_pdb_id', 'ligand_unique_ccd_code'],
    right_on=['pdb_id', 'ligand_name'],
    how='inner'
)

best_ccd = (
    merged_ccd
    .sort_values('system_ligand_validation_average_rsr', ascending=True)
    .drop_duplicates(subset=['pdb_id', 'ligand_name'], keep='first')
    .drop(columns=['ligand_name'])
)

# --- Entries without valid CCD codes: match on PDB only, take best RSR ---
non_ccd_entries = biolip_df[~biolip_df['is_ccd']]
merged_non_ccd = subset.merge(
    non_ccd_entries[['pdb_id']].drop_duplicates(),
    left_on=['entry_pdb_id'], right_on = ['pdb_id'],
    how='inner'
)
best_non_ccd = (
    merged_non_ccd
    .sort_values('ligand_num_heavy_atoms', ascending=False)
    .drop_duplicates(subset=['pdb_id'], keep='first')
)

# Combine both subsets
biolip_subset = (
    pd.concat([best_ccd, best_non_ccd], ignore_index=True)
    .drop_duplicates(subset=['basename'])
    .drop(columns=['pdb_id'])
    .reset_index(drop=True)
)

biolip_subset = biolip_subset[columns_to_select]
biolip_subset.to_csv(DATA_DIR / 'CROWN' / 'metadata' / 'biolip2.csv', index = False, float_format = '%.3f')
