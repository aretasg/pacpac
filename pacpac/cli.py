from pathlib import Path
from typing import Optional, List

import typer
import pandas as pd

from pacpac import pacpac
from pacpac.annotations import NB_WORKERS, ANARCI_SPECIES


cli = typer.Typer()

@cli.command()
def cluster(
    dataset_csv_path: Path,
    vh_aa_sequence_col_name: str,
    vl_aa_sequence_col_name: Optional[str] = None,
    scheme: Optional[str] = "imgt",
    cdr_scheme: Optional[str] = "chothia",
    num_extra_residues: Optional[int] = 2,
    paratope_residue_threshold: Optional[float] = 0.67,
    paratope_identity_threshold: Optional[float] = 0.75,
    clonotype_identity_threshold: Optional[float] = 0.72,
    structural_equivalence: Optional[bool] = True,
    perform_clonotyping: Optional[bool] = True,
    perform_paratyping: Optional[bool] = True,
    tokenize: Optional[bool] = False,
    allowed_species: Optional[List[str]] = ANARCI_SPECIES,
    cpu_count: Optional[int] = NB_WORKERS,
    sep: Optional[str] = ",",
) -> None:

    """
    Annotates and clusters by clonotype (single chain only) and/or
    paratope (single chain or both chains).
    """

    keyword_args = locals()
    optional_args = {key: keyword_args[key] for key in keyword_args if key not in {"dataset_csv_path", "sep"}}

    full_csv_path = Path(dataset_csv_path).resolve()
    usecols = [vh_aa_sequence_col_name]
    if vl_aa_sequence_col_name is not None:
        usecols.append(vl_aa_sequence_col_name)
    df = pd.read_csv(full_csv_path, sep=sep, usecols=usecols)
    df = pacpac.cluster(df=df, **optional_args)
    df.to_csv(full_csv_path.parent / (
        full_csv_path.stem + "_clustered" + full_csv_path.suffix
        )
    )


@cli.command()
def probe(
    vh_probe_sequence: str,
    dataset_csv_path: Path,
    vh_aa_sequence_col_name: str,
    vl_aa_sequence_col_name: Optional[str] = None,
    vl_probe_sequence: Optional[str] = None,
    scheme: Optional[str] = "imgt",
    cdr_scheme: Optional[str] = "chothia",
    num_extra_residues: Optional[int] = 2,
    paratope_residue_threshold: Optional[float] = 0.67,
    paratope_identity_threshold: Optional[float] = 0.75,
    clonotype_identity_threshold: Optional[float] = 0.72,
    structural_equivalence: Optional[bool] = True,
    perform_clonotyping: Optional[bool] = True,
    perform_paratyping: Optional[bool] = True,
    tokenize: Optional[bool] = False,
    allowed_species: Optional[List[str]] = ANARCI_SPECIES,
    cpu_count: Optional[int] = NB_WORKERS,
    sep: Optional[str] = ",",
) -> None:

    """
    Probe sequence for similar clonotypes (single chain only) and/or
    paratopes (single or both chains).
    """

    keyword_args = locals()
    optional_args = {key: keyword_args[key] for key in keyword_args if key not in {"dataset_csv_path", "sep"}}

    full_csv_path = Path(dataset_csv_path).resolve()
    usecols = [vh_aa_sequence_col_name]
    if vl_aa_sequence_col_name is not None:
        usecols.append(vl_aa_sequence_col_name)
    df = pd.read_csv(full_csv_path, sep=sep, usecols=usecols)
    df = pacpac.probe(df=df, **optional_args)
    df.to_csv(full_csv_path.parent / (
        full_csv_path.stem + "_probed" + full_csv_path.suffix
        )
    )
