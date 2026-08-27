from src.config import DATA_DIR

import pandas as pd
import os

os.chdir(f'{DATA_DIR}/CASF-2016/benchmark_results')

for dataset in ['crown', 'crown_xtal', 'hiqbind', 'hiqbind_xtal']:

	# 1: scoring power
	df1 = pd.read_csv(f'despot_{dataset}_scorepower.csv')
	df2 = pd.read_csv(f'korp_{dataset}_scorepower.csv')
	df3 = df1.copy().drop(columns = ['score'])
	df3['score'] = df1['score'] + df2['score']
	df3.to_csv(f'despot_combo_{dataset}_scorepower.csv', index = False)

	# 2: docking power
	df1 = pd.read_csv(f'despot_{dataset}_dockingpower.csv')
	df2 = pd.read_csv(f'korp_{dataset}_dockingpower.csv')
	df3 = df1.copy().drop(columns = ['score'])
	df3['score'] = df1['score'] + df2['score']
	df3.to_csv(f'despot_combo_{dataset}_dockingpower.csv', index = False)

	# 3: screening power
	df1 = pd.read_csv(f'despot_{dataset}_screeningpower.csv')
	df2 = pd.read_csv(f'korp_{dataset}_screeningpower.csv')
	df3 = df1.copy().drop(columns = ['score'])
	df3['score'] = df1['score'] + df2['score']
	df3.to_csv(f'despot_combo_{dataset}_screeningpower.csv', index = False)

