from pathlib import Path
from typing import Optional

import typer
import pandas as pd

from pacpac import pacpac


cli = typer.Typer()

@cli.command()
def cluster(
    dataset_csv_path: Path,
    vh_aa_sequence_col_name: str,
    vl_aa_sequence_col_name: Optional[str] = None,
    scheme: Optional[str] = "chothia",
    cdr_scheme: Optional[str] = "chothia",
    num_extra_residues: Optional[int] = 2,
    paratope_residue_threshold: Optional[float] = 0.67,
    paratope_identity_threshold: Optional[float] = 0.75,
    clonotype_identity_threshold: Optional[float] = 0.72,
    structural_equivalence: Optional[bool] = True,
    perform_clonotyping: Optional[bool] = True,
    tokenize: Optional[bool] = False,
) -> None:

    """
    Annotates and clusters by clonotype (single chain only)
    and paratope (single chain or both chains).
    """

    keyword_args = locals()
    optional_args = {key: keyword_args[key] for key in keyword_args if key not in {'dataset_csv_path'}}

    full_csv_path = Path(dataset_csv_path).resolve()
    df = pd.read_csv(full_csv_path)
    df = pacpac.cluster(df=df, **optional_args)
    df.to_csv(full_csv_path.parent / (
        full_csv_path.stem + '_clustered' + full_csv_path.suffix
        )
    )


@cli.command()
def probe(
    vh_probe_sequence: str,
    dataset_csv_path: Path,
    vh_aa_sequence_col_name: str,
    vl_aa_sequence_col_name: Optional[str] = None,
    vl_probe_sequence: Optional[str] = None,
    scheme: Optional[str] = "chothia",
    cdr_scheme: Optional[str] = "chothia",
    num_extra_residues: Optional[int] = 2,
    paratope_residue_threshold: Optional[float] = 0.67,
    paratope_identity_threshold: Optional[float] = 0.75,
    clonotype_identity_threshold: Optional[float] = 0.72,
    structural_equivalence: Optional[bool] = True,
    perform_clonotyping: Optional[bool] = True,
    tokenize: Optional[bool] = False,
) -> None:

    """
    Probe sequences in a dataframe for similar paratopes (single or both chains)
    and clonotypes (single chain only).
    """

    keyword_args = locals()
    optional_args = {key: keyword_args[key] for key in keyword_args if key not in {'dataset_csv_path'}}

    full_csv_path = Path(dataset_csv_path).resolve()
    df = pd.read_csv(full_csv_path)
    df = pacpac.probe(df=df, **optional_args)
    df.to_csv(full_csv_path.parent / (
        full_csv_path.stem + '_probed' + full_csv_path.suffix
        )
    )

