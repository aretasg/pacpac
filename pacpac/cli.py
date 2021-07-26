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
    Annotates and clusters by clonotype (single chain only) and paratope (single chain only or both chains).
    """

    full_csv_path = Path(dataset_csv_path).resolve()
    df = pd.read_csv(full_csv_path)
    df = pacpac.cluster(
        df,
        vh_aa_sequence_col_name,
        vl_aa_sequence_col_name=vl_aa_sequence_col_name,
        scheme=scheme,
        cdr_scheme=cdr_scheme,
        num_extra_residues=num_extra_residues,
        paratope_residue_threshold=paratope_residue_threshold,
        paratope_identity_threshold=paratope_identity_threshold,
        clonotype_identity_threshold=clonotype_identity_threshold,
        structural_equivalence=structural_equivalence,
        perform_clonotyping=perform_clonotyping,
        tokenize=tokenize,
        )
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
    Probe sequences in the pandas dataframe for similar paratopes (single chain only or both chains) and clonotypes (single chain only).
    """

    full_csv_path = Path(dataset_csv_path).resolve()
    df = pd.read_csv(full_csv_path)
    df = pacpac.probe(
        vh_probe_sequence,
        df,
        vh_aa_sequence_col_name,
        vl_aa_sequence_col_name=vl_aa_sequence_col_name,
        vl_probe_sequence=vl_probe_sequence,
        scheme=scheme,
        cdr_scheme=cdr_scheme,
        num_extra_residues=num_extra_residues,
        paratope_residue_threshold=paratope_residue_threshold,
        paratope_identity_threshold=paratope_identity_threshold,
        clonotype_identity_threshold=clonotype_identity_threshold,
        structural_equivalence=structural_equivalence,
        perform_clonotyping=perform_clonotyping,
        tokenize=tokenize,
        )
    df.to_csv(full_csv_path.parent / (
        full_csv_path.stem + '_probed' + full_csv_path.suffix
        )
    )

