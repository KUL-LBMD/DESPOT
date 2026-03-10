import os
from collections import Counter
import pandas as pd
import itertools
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm

from src.config import DATA_DIR
from src.atom_typing.parse_mol2 import MolConverter

def _convert_file(filename, database):
    """Convert a single protein-ligand pair. Runs in worker process."""
    converter = MolConverter()

    prot_df = converter.convert_mol2(
        DATA_DIR / database / 'processed_mol2' / 'receptor' / filename
    )
    lig_df = converter.convert_mol2(
        DATA_DIR / database / 'processed_mol2' / 'ligand' / filename
    )

    return filename, prot_df, lig_df

def count_atom_types_parallel(database: str) -> pd.DataFrame:
    """
    Count total occurrences of each atom type across all structures in a database,
    separately for protein (receptor) and ligand files.

    Parameters
    ----------
    database : str
        Name of the database subfolder under DATA_DIR.

    Returns
    -------
    pd.DataFrame
        Columns: atom_type, protein_count, ligand_count, total_occurrence,
                 local_reference_frame (from the MolConverter output if available).
    """

    n_workers = 16
    max_queued = 32

    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    ctx = mp.get_context('spawn')

    protein_counts: Counter = Counter()
    ligand_counts: Counter = Counter()
    file_list = os.listdir(DATA_DIR / database / 'processed_mol2' / 'receptor')
    num_files = len(file_list)

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        pending = set()
        file_iter = iter(file_list)

        for f in itertools.islice(file_iter, max_queued):
            pending.add(executor.submit(_convert_file, f, database))

        with tqdm(total=num_files, desc="Processing structures", unit="file") as pbar:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)

                for future in done:
                    filename, prot_df, lig_df = future.result()
                    if prot_df is not None:
                        for atype in prot_df["atom_type"].dropna():
                            protein_counts[atype] += 1

                    if lig_df is not None:
                        for atype in lig_df["atom_type"].dropna():
                            ligand_counts[atype] += 1

                pbar.update(1)

                try:
                    next_file = next(file_iter)
                    pending.add(executor.submit(_convert_file, next_file, database))
                except StopIteration:
                    pass

    # Build results DataFrame
    all_types = sorted(set(protein_counts.keys()) | set(ligand_counts.keys()))

    records = []
    for atype in all_types:
        p_count = protein_counts.get(atype, 0)
        l_count = ligand_counts.get(atype, 0)
        records.append(
            {
                "atom_type": atype,
                "protein_count": p_count,
                "ligand_count": l_count,
                "total_occurrence": p_count + l_count,
            }
        )

    counts_df = pd.DataFrame(records).sort_values(
        "total_occurrence", ascending=False
    ).reset_index(drop=True)

    return counts_df

def count_atom_types(database: str) -> pd.DataFrame:
    """
    Count total occurrences of each atom type across all structures in a database,
    separately for protein (receptor) and ligand files.

    Parameters
    ----------
    database : str
        Name of the database subfolder under DATA_DIR.

    Returns
    -------
    pd.DataFrame
        Columns: atom_type, protein_count, ligand_count, total_occurrence,
                 local_reference_frame (from the MolConverter output if available).
    """

    protein_counts: Counter = Counter()
    ligand_counts: Counter = Counter()
    file_list = os.listdir(DATA_DIR / database / 'processed_mol2' / 'receptor')
    num_files = len(file_list)

    for i, file in enumerate(file_list):
        print(file)
        filename, prot_df, lig_df = _convert_file(file, database)

        if prot_df is not None:
            for atype in prot_df["atom_type"].dropna():
                protein_counts[atype] += 1

        if lig_df is not None:
            for atype in lig_df["atom_type"].dropna():
                ligand_counts[atype] += 1

        print(i)

    # Build results DataFrame
    all_types = sorted(set(protein_counts.keys()) | set(ligand_counts.keys()))

    records = []
    for atype in all_types:
        p_count = protein_counts.get(atype, 0)
        l_count = ligand_counts.get(atype, 0)
        records.append(
            {
                "atom_type": atype,
                "protein_count": p_count,
                "ligand_count": l_count,
                "total_occurrence": p_count + l_count,
            }
        )

    counts_df = pd.DataFrame(records).sort_values(
        "total_occurrence", ascending=False
    ).reset_index(drop=True)

    return counts_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Count atom type occurrences across a protein-ligand database."
    )
    parser.add_argument("database", type=str, help="Database name (subfolder of DATA_DIR)")
    args = parser.parse_args()

    counts_df = count_atom_types_parallel(args.database)

    output_path = str(DATA_DIR / "metadata" / f"atom_type_counts_{args.database.lower()}.csv")
    counts_df.to_csv(output_path, index=False)

    print(f"\nWrote {len(counts_df)} atom types to {output_path}")
    print(counts_df.head(20).to_string())
