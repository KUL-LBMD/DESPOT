import pandas as pd
from src.atom_typing.parse_mol2 import MolConverter
import time

start = time.time()
converter = MolConverter()
end = time.time()

print(f'Initialization: {end - start}')

df = converter.convert_mol2('/media/drives/drive3/robin/plinder_v2/processed_mol2/5yl2__1__1.b_1.c__1.k_1.l_1.k.mol2')
