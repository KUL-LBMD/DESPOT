import re
import pandas as pd

def split_mol2(input_file, tmp_dir):
	"""Split multi-mol2 file into separate files"""

	with open(input_file, 'r') as f:
		lines = [line for line in f if not line.lstrip().startswith('#')]

	content = ''.join(lines)

	molecules = re.split(r'(?=@<TRIPOS>MOLECULE)', content)
	molecules = [m for m in molecules if m.strip()]  # Remove empty entries

	for mol in molecules:
		lines = mol.strip().split('\n')
		mol_name = lines[1].strip()
		safe_name = re.sub(r'[<>:"/\\|?*]', '_', mol_name)

		with open(f'{tmp_dir}/{safe_name}.mol2', 'w') as out:
			out.write(mol)

def write_pdb_line(atom_num, atom_name, res_name, res_num, x, y, z, bfac, element):
    return (
        f"ATOM  {atom_num:5d} "
        f"{atom_name:<4s}"
        f" {res_name:>3s}"
        f" A{res_num:4d}"
        f"   {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.00:6.2f} {bfac:6.2f}          "
        f"{element:>2s}"
    )

def write_pdbs(df, name_list, output_dir):
	"""
	Write PDB files with DESPOT atom scores stored as b-factors.

	Parameters
	----------
	df [pd.DataFrame]:
		- x, y, z: 3D coordinates
		- bfac: DESPOT atom score
		- label_num: molecule entry

	name_list [List[str]]: name of each ligand pose

	output_dir [Path]: path to write pdb's to
	"""

	# Loop over molecule entries
	for i, basename in enumerate(name_list):
		subset = df[df['label_num'] == i]

		atom_num = 1
		res_num = 1
		res_name = 'LIG'
		pdb_lines = []

		for row in subset.itertuples(index = False):
			x = row.x
			y = row.y
			z = row.z
			element = row.element
			atom_name = row.atom_name
			bfac = row.bfac

			pdb_lines.append(write_pdb_line(atom_num, atom_name, res_name, res_num, x, y, z, bfac, element))
			atom_num += 1

		with open(f'{output_dir}/{basename}.pdb', 'w') as out:
			out.write('\n'.join(pdb_lines))
			out.write('\nEND\n')
