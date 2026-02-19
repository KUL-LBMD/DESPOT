from src.config import DATA_DIR, SOURCE_DB_PATH

import pandas as pd
import os

df = pd.read_parquet(SOURCE_DB_PATH / 'index' / 'annotation_table.parquet')

subset1 = df[
                (df['ligand_is_ion'] == False)
                & (df['ligand_is_artifact'] == False)
                & (df['ligand_is_covalent'] == False)
                ]

print(subset1)
df = subset1

subset2 = df[
                  (df['ligand_num_unresolved_heavy_atoms'] == 0)
                & (df['entry_determination_method'] == 'X-RAY DIFFRACTION')
                & (df['entry_resolution'] <= 3.0)
                & (df['system_ligand_validation_average_rsr'] < 0.3)
                & (df['system_ligand_validation_average_rscc'] > 0.8)
                ]

print(subset2)
