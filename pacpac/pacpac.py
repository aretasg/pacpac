# Author: Aretas Gaspariunas

from typing import List, Dict, Optional

import pandas as pd
import numpy as np
from pyfiglet import figlet_format
from numba import njit
# from numba.typed import List as numbaList

from pacpac.utils import convert_to_typed_numba_dict, rename_dict_keys
from pacpac.annotations import annotate_sequence, annotations_for_df, \
    paratopes_for_df_both_chains, tokenize_and_reformat, init_pandarallel, NB_WORKERS


@njit(cache=True)
def check_clonotype(
    probe_v_gene: str,
    probe_j_gene: str,
    probe_cdr3_len: int,
    probe_cdr3_aa_seq: str,
    target_v_gene: str,
    target_j_gene: str,
    target_cdr3_len: int,
    target_cdr3_aa_seq: str,
) -> float:

    """
    Compares two VH sequences to check if clonotypes match.
    Returns sequence identity of HCDR3 if clonotype is the same i.e. VH and JH genes are the same.
    If clonotypes do not match, sequence identity == 0.
    """

    sequence_identity = 0

    if (
        probe_v_gene == target_v_gene
        and probe_j_gene == target_j_gene
        and probe_cdr3_len == target_cdr3_len
    ):

        # inverse hamming distance
        match_count = len(
            [
                True
                for res1, res2 in zip(probe_cdr3_aa_seq, target_cdr3_aa_seq)
                if res1 == res2
            ]
        )

        sequence_identity = match_count / probe_cdr3_len

    return sequence_identity


@njit(cache=True)
def cluster_by_clonotype(
    index_list: List[int],
    vh_gene_list: List[str],
    jh_gene_list: List[str],
    vh_cdr3_len_list: List[int],
    vh_cdr3_aa_list: List[str],
    identity_threshold: float,
) -> Dict[tuple, List[int]]:

    """
    Clusters sequences in the dataframe by clonotype using greedy incremental approach.
    Just a PoC - can be optimized further. The key is to know when to stop (I think).
    """

    cluster_count = 1
    cluster_dict = dict()
    k = {(1, 2): np.arange(1), (3, 4): np.arange(2)}  # dict types for numba compiler
    assigned_id_list = [-1]  # declaring a list of ints for numba compiler

    for index, cdr3_len, vh, jh, cdr3_aa in zip(
        index_list, vh_cdr3_len_list, vh_gene_list, jh_gene_list, vh_cdr3_aa_list
    ):

        if index in assigned_id_list:
            continue

        members = [
            index2
            for index2, cdr3_len2, vh2, jh2, cdr3_aa2 in zip(
                index_list,
                vh_cdr3_len_list,
                vh_gene_list,
                jh_gene_list,
                vh_cdr3_aa_list,
            )
            if check_clonotype(
                vh,
                jh,
                cdr3_len,
                cdr3_aa,
                vh2,
                jh2,
                cdr3_len2,
                cdr3_aa2,
            )
            >= identity_threshold
            and index2 not in assigned_id_list
        ]

        assigned_id_list += members
        if len(members) > 1:
            cluster_dict[(cluster_count, index)] = np.array(members)
            cluster_count += 1

    return cluster_dict


@njit(cache=True)
def check_paratope_equal_len_cdrs(
    probe_cdr1_aa_seq: str,
    probe_cdr2_aa_seq: str,
    probe_cdr3_aa_seq: str,
    probe_paratope_len: int,
    probe_paratope_dict: Dict[str, Dict[str, str]],
    target_cdr1_aa_seq: str,
    target_cdr2_aa_seq: str,
    target_cdr3_aa_seq: str,
    target_paratope_len: int,
    target_paratope_dict: Dict[str, Dict[str, str]],
) -> float:

    """
    Compares two sequences to check for paratope similarity.
    Matches sequences only with the same CDR lengths.
    Residue postions are used as provided by Parapred.
    Returns sequence identity between the two paratopes.

    Check `CL-97141` in `Pertussis_SC.csv` (IMGT numbering and North CDR definition)
    in the Richardson publication supplementary material for outliers to this assumption
    """

    sequence_identity = 0

    if (
        len(probe_cdr1_aa_seq) == len(target_cdr1_aa_seq)
        and len(probe_cdr2_aa_seq) == len(target_cdr2_aa_seq)
        and len(probe_cdr3_aa_seq) == len(target_cdr3_aa_seq)
    ):

        match_count = len(
            [
                True
                for cdr, residue_dict in probe_paratope_dict.items()
                for res_num, res in residue_dict.items()
                if res_num in target_paratope_dict[cdr]
                and res == target_paratope_dict[cdr][res_num]
            ]
        )

        sequence_identity = match_count / min(probe_paratope_len, target_paratope_len)

    return sequence_identity


@njit(cache=False)
def cluster_by_paratope(
    index_list: List[int],
    cdr1_aa_seq_list: List[str],
    cdr2_aa_seq_list: List[str],
    cdr3_aa_seq_list: List[str],
    paratope_len_list: List[int],
    paratope_dict_list: List[Dict[str, Dict[str, str]]],
    identity_threshold: float,
) -> Dict[tuple, List[int]]:

    """
    Clusters sequences in the dataframe by paratope using greedy incremental approach.
    """

    count = 1
    cluster_dict = dict()
    k = {(1, 2): np.arange(1), (3, 4): np.arange(2)}  # dict types for numba compiler
    assigned_id_list = [-1]  # declaring a list of ints for numba compiler

    for index, cdr1, cdr2, cdr3, paratope_len, paratope in zip(
        index_list,
        cdr1_aa_seq_list,
        cdr2_aa_seq_list,
        cdr3_aa_seq_list,
        paratope_len_list,
        paratope_dict_list,
    ):

        if index in set(assigned_id_list):
            continue

        members = [
            index2
            for index2, cdr1_2, cdr2_2, cdr3_2, paratope_len2, paratope2 in zip(
                index_list,
                cdr1_aa_seq_list,
                cdr2_aa_seq_list,
                cdr3_aa_seq_list,
                paratope_len_list,
                paratope_dict_list,
            )
            if index2 not in set(assigned_id_list)
            and check_paratope_equal_len_cdrs(
                cdr1,
                cdr2,
                cdr3,
                paratope_len,
                paratope,
                cdr1_2,
                cdr2_2,
                cdr3_2,
                paratope_len2,
                paratope2,
            )
            >= identity_threshold
        ]

        assigned_id_list += members
        if len(members) > 1:
            cluster_dict[(count, index)] = np.array(members)
            count += 1

    return cluster_dict


@njit(cache=True)
def check_paratope_structural(
    probe_paratope_len: int,
    probe_paratope_dict: Dict[str, Dict[str, str]],
    target_paratope_len: int,
    target_paratope_dict: Dict[str, Dict[str, str]],
) -> float:

    """
    Compares two sequences to check for paratope similarity using structural equivalence.
    Uses structural equivalence provided by the numbering scheme and compares different length CDRs.
    Divides the the number of paratope matches by the longer paratope to be
        more sensitive to paratope residue count mismatches.
    Returns sequence identity between the two paratopes.
    """

    match_count = len(
        [
            True
            for cdr, residue_dict in probe_paratope_dict.items()
            for res_num, res in residue_dict.items()
            if res_num in target_paratope_dict[cdr]
            and res == target_paratope_dict[cdr][res_num]
        ]
    )

    sequence_identity = match_count / max(probe_paratope_len, target_paratope_len)

    return sequence_identity


@njit(cache=False)
def cluster_by_paratope_structural(
    index_list: List[int],
    paratope_len_list: List[int],
    paratope_dict_list: List[Dict[str, Dict[str, str]]],
    identity_threshold: float,
) -> Dict[tuple, List[int]]:

    """
    Clusters sequences in the dataframe by paratope using greedy incremental approach.
    """

    count = 1
    cluster_dict = dict()
    k = {(1, 2): np.arange(1), (3, 4): np.arange(2)}  # dict types for numba compiler
    assigned_id_list = [-1]  # declaring a list of ints for numba compiler

    for index, paratope_len, paratope in zip(
        index_list,
        paratope_len_list,
        paratope_dict_list,
    ):

        if index in set(assigned_id_list):
            continue

        members = [
            index2
            for index2, paratope_len2, paratope2 in zip(
                index_list,
                paratope_len_list,
                paratope_dict_list,
            )
            if index2 not in set(assigned_id_list)
            and check_paratope_structural(
                paratope_len,
                paratope,
                paratope_len2,
                paratope2,
            )
            >= identity_threshold
        ]

        assigned_id_list += members
        if len(members) > 1:
            cluster_dict[(count, index)] = np.array(members)
            count += 1

    return cluster_dict


def start_and_checks(scheme: str, cdr_scheme: str) -> bool:

    """
    Prints starting message to the console and checks scheme and cdr_scheme inputs
    """

    print(figlet_format("PaCPaC", font="banner"))

    scheme = scheme.lower()
    if scheme not in ("chothia", "imgt", "martin"):
        print(
            f"You don't want to input *{scheme}*. You want to go home and rethink your life"
        )
        return False

    cdr_scheme = cdr_scheme.lower()
    if cdr_scheme not in ("chothia", "imgt", "north", "contact"):
        print(
            f"You don't want to input *{cdr_scheme}*. You want to go home and rethink your life"
        )
        return False

    return True


def cluster(
    df: pd.DataFrame,
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
    allowed_species: Optional[List[str]] = None,
    cpu_count: Optional[int] = NB_WORKERS
) -> pd.DataFrame:

    """
    Annotates and clusters by clonotype (single chain only) and paratope (single chain only or both chains).

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe to cluster.
    vh_aa_sequence_col_name : str
        column name for VH sequences to cluster.
    vl_aa_sequence_col_name : str, None
        column name for VL sequences to cluster.
    scheme : str, default "imgt"
        numbering scheme to use. IMGT, Chothia, Martin only.
    cdr_scheme : str, default "chothia"
        CDR definition to use. IMGT, North, Chothia and Contact supported.
        IMGT, North can be used with IMGT numbering scheme.
        North, Chothia, Contact can be used with Chothia, Martin numbering schemes.
        CDR definition with not supported numbering scheme will default to the supported numbering scheme.
    num_extra_residues : int, default 2
        include extra residues at the start and end of each CDR.
    paratope_residue_threshold : float, default 0.67
        paratope residue call threshold for parapred predictions.
    paratope_identity_threshold : float, default 0.75
        paratope sequence identity value at which paratope are considered the same.
    clonotype_identity_threshold : float, default 0.72
        clonotype sequence identity value at which clonotype are considered the same.
    structural_equivalence : bool, default True
        specify whether structural equivalence as assigned by the numbering scheme of
        choice should be used and paratopes with different CDR lengths to be compared.
        Always defaults to True if clustering VH and VL pairs.
    perform_clonotyping : bool, default True
        specify if clonotyping should be performed.
    perform_paratyping : bool, default True
        specify if paratyping should be performed.
    tokenize : bool, default False
        specify if residues should be tokenized to match residues of the same group when paratope clustering.
    allowed_species : List[str], default None
        limit the species that can be assigned with ANARCI to a limited set. Useful when the
        animal species is known or when performing closest germline experiments. Choose a subset of
        ['human', 'mouse', 'rat', 'rabbit', 'rhesus', 'pig', 'alpaca']. Default is to use all species.
    cpu_count : int, default - use all available CPU cores
        specify how many CPU cores should be used for parallelizing processes

    Returns
    -------
    pd.DataFrame
    """

    status = start_and_checks(scheme, cdr_scheme)
    if status is False:
        return None

    both_chains = False
    if vl_aa_sequence_col_name is not None:
        both_chains = True
        if structural_equivalence is False:
            print("structural equivalence set to True")
            structural_equivalence = True
        print("Always two there are, no more, no less. A VH and a VL")

    # slicing rows where both VH and VL is NaN
    if both_chains:
        nan_df = df[
            df[vh_aa_sequence_col_name].isnull() & df[vl_aa_sequence_col_name].isnull()
        ]
        df = df[
            df[vh_aa_sequence_col_name].notnull()
            & df[vl_aa_sequence_col_name].notnull()
        ]
    else:
        nan_df = df[df[vh_aa_sequence_col_name].isnull()]
        df = df[df[vh_aa_sequence_col_name].notnull()]
    if not nan_df.empty:
        print("Not to worry, we are still flying half a dataset")

    print("I'll try annotating, that's a good trick")
    init_pandarallel(nb_workers=cpu_count)
    df = annotations_for_df(
        df,
        vh_aa_sequence_col_name,
        assign_germline=perform_clonotyping,
        scheme=scheme,
        cdr_scheme=cdr_scheme,
        num_extra_residues=num_extra_residues,
        allowed_species=allowed_species
    )

    vl_df = None
    if both_chains:
        # annotating VL sequences
        vl_df = annotations_for_df(
            df[[vl_aa_sequence_col_name]],
            vl_aa_sequence_col_name,
            assign_germline=False,
            scheme=scheme,
            cdr_scheme=cdr_scheme,
            num_extra_residues=num_extra_residues,
            allowed_species=allowed_species
        )
        # exclude sequences where anarci has failed
        nan_df2 = df[df["CDR3"].isnull() | vl_df["CDR3"].isnull()]
        df = df[df["CDR3"].notnull() & vl_df["CDR3"].notnull()]
        vl_df = vl_df[df["CDR3"].notnull() & vl_df["CDR3"].notnull()]
    else:
        nan_df2 = df[df["CDR3"].isnull()]
        df = df[df["CDR3"].notnull()]

    if df.empty is True:
        raise ValueError('ANARCI failed to number the sequences. \
            Make sure amino acid sequences are used as input!')

    def assign_cluster(
        df: pd.DataFrame,
        cluster_dict: Dict[str, List[int]],
        cluster_type: str
    ) -> pd.DataFrame:

        if not cluster_dict:
            df[f'{cluster_type}_CLUSTER'] = None
            return df

        cluster_df = pd.DataFrame.from_dict(
            cluster_dict,
            orient='index'
            ).stack().to_frame().reset_index().drop('level_1', axis=1)
        cluster_df.columns = [f'{cluster_type}_CLUSTER', 'id']
        cluster_df[f'{cluster_type}_CLUSTER'] = [
            str(i[0]) + f" (seed: {i[1]})"
            for i in cluster_df[f'{cluster_type}_CLUSTER']
            ]
        cluster_df.set_index('id', inplace=True)

        return pd.merge(df, cluster_df, how='left', left_index=True, right_index=True)


    if perform_clonotyping is True:

        df["HCDR1_LEN"] = df["CDR1"].astype(str).map(len)
        df["HCDR2_LEN"] = df["CDR2"].astype(str).map(len)
        df["HCDR3_LEN"] = df["CDR3"].astype(str).map(len)

        # exclude sequences 0 len for HCDR3
        nan_df3 = df[df['HCDR3_LEN'].values <= num_extra_residues * 2]
        df = df[df['HCDR3_LEN'].values > num_extra_residues * 2]

        df.sort_values(
            ["HCDR3_LEN", "HCDR2_LEN", "HCDR1_LEN", vh_aa_sequence_col_name],
            ascending=False,
            inplace=True,
        )

        print("Now THIS is clonotype clustering")
        end_cdr = -1 * num_extra_residues
        if num_extra_residues == 0:
            end_cdr = None

        clonotype_cluster_dict = cluster_by_clonotype(
            df.index.tolist(),
            df["V_GENE"].tolist(),
            df["J_GENE"].tolist(),
            (df["HCDR3_LEN"] - 2 * num_extra_residues).tolist(),
            df["CDR3"].str[num_extra_residues:end_cdr].tolist(),
            clonotype_identity_threshold,
        )

        df = assign_cluster(df, clonotype_cluster_dict, 'CLONOTYPE')

        df = pd.concat(
            [df, nan_df3], ignore_index=False, axis=0, sort=False
            )

        print("Your clonotypes are very impressive. You must be very proud")

    if perform_paratyping is True:

        print("Learning to stop worrying and falling in love with the paratope")

        df, nan_df4 = paratopes_for_df_both_chains(
            df,
            vl_df,
            both_chains=both_chains,
            paratope_residue_threshold=paratope_residue_threshold,
        )

        print("Hold on. This whole paratope clustering was your idea")

        df = tokenize_and_reformat(df, structural_equivalence=True, tokenize=tokenize)

        df.sort_values(
            ["PARATOPE_LEN", vh_aa_sequence_col_name],
            ascending=False,
            inplace=True,
        )

        if structural_equivalence is False and both_chains is False:
            paratope_cluster_dict = cluster_by_paratope(
                df.index.tolist(),
                df["CDR1"].tolist(),
                df["CDR2"].tolist(),
                df["CDR3"].tolist(),
                df["PARATOPE_LEN"].tolist(),
                df["PARATOPE_DICT_REFORMAT"].tolist(),
                paratope_identity_threshold,
            )
        else:
            paratope_cluster_dict = cluster_by_paratope_structural(
                df.index.tolist(),
                df["PARATOPE_LEN"].tolist(),
                df["PARATOPE_DICT_REFORMAT"].tolist(),
                paratope_identity_threshold,
            )

        df = assign_cluster(df, paratope_cluster_dict, 'PARATOPE')
        print("200,000 paratopes are ready, with a million more well on the way")

        df = pd.concat(
            [df, nan_df4], ignore_index=False, axis=0, sort=False
            )

    df = pd.concat(
        [df, nan_df, nan_df2], ignore_index=False, axis=0, sort=False
    )
    df.drop(
        [
            "CDR1_NUMBERING",
            "CDR2_NUMBERING",
            "CDR3_NUMBERING",
            "PARATOPE_DICT",
            "PARATOPE_DICT_NUMBERED",
            "PARATOPE_DICT_REFORMAT",
            "PARATOPE_PROBS",
            "PARATOPE_PROBS_NUMBERED",
            "LPARATOPE_DICT",
            "LPARATOPE_DICT_NUMBERED",
            "LPARATOPE",
            "CHAIN_TYPE",
        ],
        axis=1,
        inplace=True,
        errors="ignore",
    )
    if both_chains:
        df.drop(
            ["CDR1", "CDR2", "CDR3"],
            axis=1,
            inplace=True,
            errors="ignore",
    )

    df.sort_index(inplace=True)

    print("Another happy clustering")

    return df


def probe(
    vh_probe_sequence: str,
    df: pd.DataFrame,
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
    allowed_species: Optional[List[str]] = None,
    cpu_count: Optional[int] = NB_WORKERS
) -> pd.DataFrame:

    """
    Probe sequences in the pandas dataframe for similar paratopes (single chain only or both chains)
    and clonotypes (single chain only).

    Parameters
    ----------
    vh_probe_sequence : str
        VH sequence to use as a probe.
    df : pd.DataFrame
        pandas dataframe to probe.
    vh_aa_sequence_col_name : str
        column name for VH sequences to probe.
    vl_aa_sequence_col_name : str, None
        column name for VL sequences to probe.
    vl_probe_sequence : str, None
        VL sequence to use as a probe.
    scheme : str, default "imgt"
        numbering scheme to use.
    cdr_scheme : str, default "chothia"
        CDR definition to use. IMGT, North, Chothia and Contact supported.
        IMGT, North can be used with IMGT numbering scheme.
        North, Chothia, Contact can be used with Chothia, Martin numbering schemes.
        CDR definition with not supported numbering scheme will default to the supported numbering scheme.
    num_extra_residues : int, default 2
        include extra residues at the start and end of each CDR.
    paratope_residue_threshold : float, default 0.67
        paratope residue call threshold for parapred predictions.
    paratope_identity_threshold : float, default 0.75
        paratope sequence identity value at which paratope are considered the same.
    clonotype_identity_threshold : float, default 0.72
        clonotype sequence identity value at which clonotype are considered the same.
    structural_equivalence : bool, default True
        specify whether structural equivalence as assigned by the numbering scheme of
        choice should be used and paratopes with different CDR lengths to be compared.
        Always defaults to True if clustering VH and VL pairs.
    perform_clonotyping : bool, default True
        specify if clonotyping should be performed.
    tokenize : bool, default False
        specify if residues should be tokenized to match residues of the same group when paratope probing.
    allowed_species : List[str], default None
        limit the species that can be assigned with ANARCI to a limited set. Useful when the
        animal species is known or when performing closest germline experiments. Choose a subset of
        ['human', 'mouse', 'rat', 'rabbit', 'rhesus', 'pig', 'alpaca']. Default is to use all species.
    cpu_count : int, default - use all available CPU cores
        specify how many CPU cores should be used for parallelizing processes

    Returns
    -------
    pd.DataFrame
    """

    status = start_and_checks(scheme, cdr_scheme)
    if status is False:
        return None

    both_chains = False
    if vl_probe_sequence is not None and vl_aa_sequence_col_name is not None:
        both_chains = True
        if structural_equivalence is False:
            print("structural equivalence set to True")
        print("Always two there are, no more, no less. A VH and a VL")

    # annotations for the probe
    probe_dict = annotate_sequence(
        vh_probe_sequence,
        scheme=scheme,
        cdr_scheme=cdr_scheme,
        assign_germline=perform_clonotyping,
        num_extra_residues=num_extra_residues,
        paratope_residue_threshold=paratope_residue_threshold,
        structural_equivalence=structural_equivalence,
        tokenize=tokenize,
        allowed_species=allowed_species,
    )
    if both_chains:
        vl_probe_dict = annotate_sequence(
            vl_probe_sequence,
            scheme=scheme,
            cdr_scheme=cdr_scheme,
            assign_germline=False,
            num_extra_residues=num_extra_residues,
            paratope_residue_threshold=paratope_residue_threshold,
            structural_equivalence=True,
            tokenize=tokenize,
            allowed_species=allowed_species
        )

        # merge probe dicts
        probe_dict["PARATOPE_DICT_REFORMAT"] = rename_dict_keys(
            probe_dict["PARATOPE_DICT_REFORMAT"], "H"
        )
        vl_probe_dict["PARATOPE_DICT_REFORMAT"] = rename_dict_keys(
            vl_probe_dict["PARATOPE_DICT_REFORMAT"], "L"
        )
        probe_dict["PARATOPE_DICT_REFORMAT"] = {
            **probe_dict["PARATOPE_DICT_REFORMAT"],
            **vl_probe_dict["PARATOPE_DICT_REFORMAT"],
        }
        probe_dict["PARATOPE_DICT_REFORMAT"] = convert_to_typed_numba_dict(
            probe_dict["PARATOPE_DICT_REFORMAT"]
        )
        # slicing rows where both VH and VL is NaN
        nan_df = df[
            df[vh_aa_sequence_col_name].isnull() & df[vl_aa_sequence_col_name].isnull()
        ]
        df = df[
            df[vh_aa_sequence_col_name].notnull()
            & df[vl_aa_sequence_col_name].notnull()
        ]
    else:
        nan_df = df[df[vh_aa_sequence_col_name].isnull()]
        df = df[df[vh_aa_sequence_col_name].notnull()]
    if not nan_df.empty:
        print("Not to worry, we are still flying half a dataset")

    # annotate sequences with genes and cdrs
    print("I'll try annotating, that's a good trick")
    init_pandarallel(nb_workers=cpu_count)
    df = annotations_for_df(
        df,
        vh_aa_sequence_col_name,
        assign_germline=perform_clonotyping,
        scheme=scheme,
        cdr_scheme=cdr_scheme,
        num_extra_residues=num_extra_residues,
        allowed_species=allowed_species
    )

    vl_df = None
    if both_chains:
        # annotating VL sequences
        vl_df = annotations_for_df(
            df[[vl_aa_sequence_col_name]],
            vl_aa_sequence_col_name,
            assign_germline=False,
            scheme=scheme,
            cdr_scheme=cdr_scheme,
            num_extra_residues=num_extra_residues,
            allowed_species=allowed_species
        )

        # exclude sequences where anarci has failed
        nan_df2 = df[df["CDR3"].isnull() | vl_df["CDR3"].isnull()]
        df = df[df["CDR3"].notnull() & vl_df["CDR3"].notnull()]
        vl_df = vl_df[df["CDR3"].notnull() & vl_df["CDR3"].notnull()]
    else:
        nan_df2 = df[df["CDR3"].isnull()]
        df = df[df["CDR3"].notnull()]

    if df.empty is True:
        raise ValueError('ANARCI failed to number the sequences. \
            Make sure amino acid sequences are used as input!')

    # probing with clonotype
    if perform_clonotyping is True:

        df["HCDR1_LEN"] = df["CDR1"].astype(str).map(len)
        df["HCDR2_LEN"] = df["CDR2"].astype(str).map(len)
        df["HCDR3_LEN"] = df["CDR3"].astype(str).map(len)

        # exclude sequences 0 len for HCDR3
        nan_df3 = df[df['HCDR3_LEN'].values <= num_extra_residues * 2]
        df = df[df['HCDR3_LEN'].values > num_extra_residues * 2]

        probe_cdr3_len = len(probe_dict["CDR3"]) - 2 * num_extra_residues
        if probe_cdr3_len == 0:
            raise ValueError("Probe CDR3 length cannot be zero")

        print("We're just clonotypes, sir. We're meant to be expendable")

        end_cdr = -1 * num_extra_residues
        if num_extra_residues == 0:
            end_cdr = None

        df["CLONOTYPE_MATCH"] = [
            check_clonotype(
                probe_dict["V_GENE"],
                probe_dict["J_GENE"],
                probe_cdr3_len,
                probe_dict["CDR3"][num_extra_residues:end_cdr],
                vh,
                jh,
                cdr3_len,
                cdr3_aa,
            )
            for vh, jh, cdr3_len, cdr3_aa in zip(
                df["V_GENE"],
                df["J_GENE"],
                (df["HCDR3_LEN"] - 2 * num_extra_residues),
                df["CDR3"].str[num_extra_residues:end_cdr]
            )
        ]

        df = pd.concat(
            [df, nan_df3], ignore_index=False, axis=0, sort=False
            )
    else:
        df["CLONOTYPE_MATCH"] = 0

    if perform_paratyping is True:

        # create paratopes for sequences in df
        print("Learning to stop worrying and falling in love with the paratope")

        df, nan_df4 = paratopes_for_df_both_chains(
            df,
            vl_df,
            both_chains=both_chains,
            paratope_residue_threshold=paratope_residue_threshold,
        )
        df = tokenize_and_reformat(
            df, structural_equivalence=structural_equivalence, tokenize=tokenize
        )

        # probing with paratope
        print("This is where the paratope probing begins")
        if structural_equivalence is False and both_chains is False:
            df["PARATOPE_MATCH"] = [
                check_paratope_equal_len_cdrs(
                    probe_dict["CDR1"],
                    probe_dict["CDR2"],
                    probe_dict["CDR3"],
                    probe_dict["PARATOPE_LEN"],
                    probe_dict["PARATOPE_DICT_REFORMAT"],
                    cdr1,
                    cdr2,
                    cdr3,
                    paratope_len,
                    paratope,
                )
                for cdr1, cdr2, cdr3, paratope_len, paratope in zip(
                    df["CDR1"],
                    df["CDR2"],
                    df["CDR3"],
                    df["PARATOPE_LEN"],
                    df["PARATOPE_DICT_REFORMAT"],
                )
            ]
        else:
            df["PARATOPE_MATCH"] = [
                check_paratope_structural(
                    probe_dict["PARATOPE_LEN"],
                    probe_dict["PARATOPE_DICT_REFORMAT"],
                    paratope_len,
                    paratope,
                )
                for paratope_len, paratope in zip(
                    df["PARATOPE_LEN"],
                    df["PARATOPE_DICT_REFORMAT"],
                )
            ]

        df = pd.concat(
            [df, nan_df4], ignore_index=False, axis=0, sort=False
            )
    else:
        df["PARATOPE_MATCH"] = 0

    df["PREDICTION_SPACE"] = [
        "both"
        if p_match >= paratope_identity_threshold and c_match >= clonotype_identity_threshold
        else "paratope-only"
        if p_match >= paratope_identity_threshold
        else "clonotype-only"
        if c_match >= clonotype_identity_threshold
        else None
        for p_match, c_match in zip(df["PARATOPE_MATCH"], df["CLONOTYPE_MATCH"])
    ]

    df = pd.concat(
        [df, nan_df, nan_df2], ignore_index=False, axis=0, sort=False
    )
    df.drop(
        [
            "CDR1_NUMBERING",
            "CDR2_NUMBERING",
            "CDR3_NUMBERING",
            "PARATOPE_PROBS",
            "PARATOPE_PROBS_NUMBERED",
            "PARATOPE_DICT",
            "PARATOPE_DICT_NUMBERED",
            "PARATOPE_DICT_REFORMAT",
            "LPARATOPE_DICT",
            "LPARATOPE_DICT_NUMBERED",
            "LPARATOPE",
            "CHAIN_TYPE",
        ],
        axis=1,
        inplace=True,
        errors="ignore",
    )
    if both_chains:
        df.drop(
            ["CDR1", "CDR2", "CDR3"],
            axis=1,
            inplace=True,
            errors="ignore",
    )

    print("Another happy probing")

    return df


def probe_multiple(
    probe_df: pd.DataFrame,
    df: pd.DataFrame,
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
    allowed_species: Optional[List[str]] = None,
    cpu_count: Optional[int] = NB_WORKERS
) -> pd.DataFrame:

    """
    Probe sequences in the pandas dataframe for similar paratopes (single chain only or both chains)
    and clonotypes (single chain only).

    Parameters
    ----------
    probe_df : pd.DataFrame
        probe sequences.
    df : pd.DataFrame
        pandas dataframe to probe.
    vh_aa_sequence_col_name : str
        column name for VH sequences to probe.
    vl_aa_sequence_col_name : str, None
        column name for VL sequences to probe.
    scheme : str, default "imgt"
        numbering scheme to use.
    cdr_scheme : str, default "chothia"
        CDR definition to use. IMGT, North, Chothia and Contact supported.
        IMGT, North can be used with IMGT numbering scheme.
        North, Chothia, Contact can be used with Chothia, Martin numbering schemes.
        CDR definition with not supported numbering scheme will default to the supported numbering scheme.
    num_extra_residues : int, default 2
        include extra residues at the start and end of each CDR.
    paratope_residue_threshold : float, default 0.67
        paratope residue call threshold for parapred predictions.
    paratope_identity_threshold : float, default 0.75
        paratope sequence identity value at which paratope are considered the same.
    clonotype_identity_threshold : float, default 0.72
        clonotype sequence identity value at which clonotype are considered the same.
    structural_equivalence : bool, default True
        specify whether structural equivalence as assigned by the numbering scheme of
        choice should be used and paratopes with different CDR lengths to be compared.
        Always defaults to True if clustering VH and VL pairs.
    perform_clonotyping : bool, default True
        specify if clonotyping should be performed.
    tokenize : bool, default False
        specify if residues should be tokenized to match residues of the same group when paratope probing.
    allowed_species : List[str], default None
        limit the species that can be assigned with ANARCI to a limited set. Useful when the
        animal species is known or when performing closest germline experiments. Choose a subset of
        ['human', 'mouse', 'rat', 'rabbit', 'rhesus', 'pig', 'alpaca']. Default is to use all species.
    cpu_count : int, default - use all available CPU cores
        specify how many CPU cores should be used for parallelizing processes

    Returns
    -------
    pd.DataFrame
    """

    status = start_and_checks(scheme, cdr_scheme)
    if status is False:
        return None

    both_chains = False
    if vl_aa_sequence_col_name is not None:
        both_chains = True
        if structural_equivalence is False:
            print("structural equivalence set to True")
            structural_equivalence = True
        print("Always two there are, no more, no less. A VH and a VL")

    # annotations for the probes
    final_probe_dict = {}
    probe_df.reset_index(inplace=True)
    for index, row in probe_df.iterrows():
        probe_dict = annotate_sequence(
            row[vh_aa_sequence_col_name],
            scheme=scheme,
            cdr_scheme=cdr_scheme,
            assign_germline=perform_clonotyping,
            num_extra_residues=num_extra_residues,
            paratope_residue_threshold=paratope_residue_threshold,
            structural_equivalence=structural_equivalence,
            tokenize=tokenize,
            allowed_species=allowed_species,
        )
        if both_chains:
            vl_probe_dict = annotate_sequence(
                row[vl_aa_sequence_col_name],
                scheme=scheme,
                cdr_scheme=cdr_scheme,
                assign_germline=False,
                num_extra_residues=num_extra_residues,
                paratope_residue_threshold=paratope_residue_threshold,
                structural_equivalence=True,
                tokenize=tokenize,
                allowed_species=allowed_species
            )

            # merge probe dicts
            probe_dict["PARATOPE_DICT_REFORMAT"] = rename_dict_keys(
                probe_dict["PARATOPE_DICT_REFORMAT"], "H"
            )
            vl_probe_dict["PARATOPE_DICT_REFORMAT"] = rename_dict_keys(
                vl_probe_dict["PARATOPE_DICT_REFORMAT"], "L"
            )
            probe_dict["PARATOPE_DICT_REFORMAT"] = {
                **probe_dict["PARATOPE_DICT_REFORMAT"],
                **vl_probe_dict["PARATOPE_DICT_REFORMAT"],
            }
            probe_dict["PARATOPE_DICT_REFORMAT"] = convert_to_typed_numba_dict(
                probe_dict["PARATOPE_DICT_REFORMAT"]
            )
        final_probe_dict[row['index']] = probe_dict

    annotated_probe_df = pd.DataFrame.from_dict(final_probe_dict, orient='index')
    annotated_probe_df.drop(
        [
            "CDR1_NUMBERING",
            "CDR2_NUMBERING",
            "CDR3_NUMBERING",
            "PARATOPE_PROBS",
            "PARATOPE_PROBS_NUMBERED",
            "PARATOPE_DICT",
            "PARATOPE_DICT_NUMBERED",
            "PARATOPE_DICT_REFORMAT",
            "LPARATOPE_DICT",
            "LPARATOPE_DICT_NUMBERED",
            "LPARATOPE",
            "CHAIN_TYPE",
        ],
        axis=1,
        inplace=True,
        errors="ignore",
    )
    if both_chains:
        annotated_probe_df.drop(
            ["CDR1", "CDR2", "CDR3"],
            axis=1,
            inplace=True,
            errors="ignore",
    )
    annotated_probe_df.to_csv('probes_annotated.csv')

    # slicing rows where both VH and VL is NaN
    if both_chains:
        nan_df = df[
            df[vh_aa_sequence_col_name].isnull() & df[vl_aa_sequence_col_name].isnull()
        ]
        df = df[
            df[vh_aa_sequence_col_name].notnull()
            & df[vl_aa_sequence_col_name].notnull()
        ]
    else:
        nan_df = df[df[vh_aa_sequence_col_name].isnull()]
        df = df[df[vh_aa_sequence_col_name].notnull()]
    if not nan_df.empty:
        print("Not to worry, we are still flying half a dataset")

    # annotate sequences with genes and cdrs
    print("I'll try annotating, that's a good trick")
    init_pandarallel(nb_workers=cpu_count)
    df = annotations_for_df(
        df,
        vh_aa_sequence_col_name,
        assign_germline=perform_clonotyping,
        scheme=scheme,
        cdr_scheme=cdr_scheme,
        num_extra_residues=num_extra_residues,
        allowed_species=allowed_species
    )

    vl_df = None
    if both_chains:
        # annotating VL sequences
        vl_df = annotations_for_df(
            df[[vl_aa_sequence_col_name]],
            vl_aa_sequence_col_name,
            assign_germline=False,
            scheme=scheme,
            cdr_scheme=cdr_scheme,
            num_extra_residues=num_extra_residues,
            allowed_species=allowed_species
        )

        # exclude sequences where anarci has failed
        nan_df2 = df[df["CDR3"].isnull() | vl_df["CDR3"].isnull()]
        df = df[df["CDR3"].notnull() & vl_df["CDR3"].notnull()]
        vl_df = vl_df[df["CDR3"].notnull() & vl_df["CDR3"].notnull()]
    else:
        nan_df2 = df[df["CDR3"].isnull()]
        df = df[df["CDR3"].notnull()]

    if df.empty is True:
        raise ValueError('ANARCI failed to number the sequences. \
            Make sure amino acid sequences are used as input!')

    # probing with clonotype
    if perform_clonotyping is True:

        df["HCDR1_LEN"] = df["CDR1"].astype(str).map(len)
        df["HCDR2_LEN"] = df["CDR2"].astype(str).map(len)
        df["HCDR3_LEN"] = df["CDR3"].astype(str).map(len)

        # exclude sequences 0 len for HCDR3
        nan_df3 = df[df['HCDR3_LEN'].values <= num_extra_residues * 2]
        df = df[df['HCDR3_LEN'].values > num_extra_residues * 2]

        for index, probe_dict in final_probe_dict.items():
            probe_cdr3_len = len(probe_dict["CDR3"]) - 2 * num_extra_residues
            if probe_cdr3_len == 0:
                raise ValueError("Probe CDR3 length cannot be zero")

        print("We're just clonotypes, sir. We're meant to be expendable")

        end_cdr = -1 * num_extra_residues
        if num_extra_residues == 0:
            end_cdr = None

        for index, probe_dict in final_probe_dict.items():
            df[f"CLONOTYPE_MATCH_{index}"] = [
                check_clonotype(
                    probe_dict["V_GENE"],
                    probe_dict["J_GENE"],
                    probe_cdr3_len,
                    probe_dict["CDR3"][num_extra_residues:end_cdr],
                    vh,
                    jh,
                    cdr3_len,
                    cdr3_aa,
                )
                for vh, jh, cdr3_len, cdr3_aa in zip(
                    df["V_GENE"],
                    df["J_GENE"],
                    (df["HCDR3_LEN"] - 2 * num_extra_residues),
                    df["CDR3"].str[num_extra_residues:end_cdr]
                )
            ]

        df = pd.concat(
            [df, nan_df3], ignore_index=False, axis=0, sort=False
            )

    if perform_paratyping is True:

        # create paratopes for sequences in df
        print("Learning to stop worrying and falling in love with the paratope")

        df, nan_df4 = paratopes_for_df_both_chains(
            df,
            vl_df,
            both_chains=both_chains,
            paratope_residue_threshold=paratope_residue_threshold,
        )
        df = tokenize_and_reformat(
            df, structural_equivalence=structural_equivalence, tokenize=tokenize
        )

        # probing with paratope
        print("This is where the paratope probing begins")
        if structural_equivalence is False and both_chains is False:
            for index, probe_dict in final_probe_dict.items():
                df[f"PARATOPE_MATCH_{index}"] = [
                    check_paratope_equal_len_cdrs(
                        probe_dict["CDR1"],
                        probe_dict["CDR2"],
                        probe_dict["CDR3"],
                        probe_dict["PARATOPE_LEN"],
                        probe_dict["PARATOPE_DICT_REFORMAT"],
                        cdr1,
                        cdr2,
                        cdr3,
                        paratope_len,
                        paratope,
                    )
                    for cdr1, cdr2, cdr3, paratope_len, paratope in zip(
                        df["CDR1"],
                        df["CDR2"],
                        df["CDR3"],
                        df["PARATOPE_LEN"],
                        df["PARATOPE_DICT_REFORMAT"],
                    )
            ]
        else:
            for index, probe_dict in final_probe_dict.items():
                df[f"PARATOPE_MATCH_{index}"] = [
                    check_paratope_structural(
                        probe_dict["PARATOPE_LEN"],
                        probe_dict["PARATOPE_DICT_REFORMAT"],
                        paratope_len,
                        paratope,
                    )
                    for paratope_len, paratope in zip(
                        df["PARATOPE_LEN"],
                        df["PARATOPE_DICT_REFORMAT"],
                    )
                ]

        df = pd.concat(
            [df, nan_df4], ignore_index=False, axis=0, sort=False
            )

    df = pd.concat(
        [df, nan_df, nan_df2], ignore_index=False, axis=0, sort=False
    )
    df.drop(
        [
            "CDR1_NUMBERING",
            "CDR2_NUMBERING",
            "CDR3_NUMBERING",
            "PARATOPE_PROBS",
            "PARATOPE_PROBS_NUMBERED",
            "PARATOPE_DICT",
            "PARATOPE_DICT_NUMBERED",
            "PARATOPE_DICT_REFORMAT",
            "LPARATOPE_DICT",
            "LPARATOPE_DICT_NUMBERED",
            "LPARATOPE",
            "CHAIN_TYPE",
        ],
        axis=1,
        inplace=True,
        errors="ignore",
    )
    if both_chains:
        df.drop(
            ["CDR1", "CDR2", "CDR3"],
            axis=1,
            inplace=True,
            errors="ignore",
    )

    print("Another happy probing")

    return df
