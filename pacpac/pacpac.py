#!/usr/bin/env python

# Author: Aretas Gaspariunas

from typing import List, Dict, Optional, Union

import pandas as pd
from pandarallel import pandarallel
from anarci import anarci
from pyfiglet import figlet_format

from pacpac.parapred.parapred import predict_sequence_probabilities as parapred

# disable TF messages
import warnings
import os

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# disable Keras messages
from contextlib import redirect_stderr

with redirect_stderr(open(os.devnull, "w")):
    import keras

pandarallel.initialize(verbose=1)


def run_and_parse_anarci(
    sequence: str,
    allow: Optional[set] = {"H", "K", "L"},
    scheme: Optional[str] = "imgt",
    assign_germline: Optional[bool] = True
) -> Dict[str, str]:

    """
    Finds V and J germline genes and assigns numbering using anarci for a given amino acid sequence.
    """

    anarci_output = anarci(
        [("1", sequence)],
        scheme=scheme,
        assign_germline=assign_germline,
        output=False,
        allow=allow,
    )
    if assign_germline:
        output_dict = {
            "CHAIN_TYPE": anarci_output[1][0][0]["chain_type"],
            "IDENTITY_SPECIES": anarci_output[1][0][0]["germlines"]["v_gene"][0][0],
            "V_GENE": anarci_output[1][0][0]["germlines"]["v_gene"][0][1],
            "V_IDENTITY": round(anarci_output[1][0][0]["germlines"]["v_gene"][1], 2),
            "J_GENE": anarci_output[1][0][0]["germlines"]["j_gene"][0][1],
            "J_IDENTITY": round(anarci_output[1][0][0]["germlines"]["j_gene"][1], 2),
            "NUMBERING": anarci_output[0][0][0][0],
        }
    else:
        output_dict = {
            "CHAIN_TYPE": anarci_output[1][0][0]["chain_type"],
            "IDENTITY_SPECIES": None,
            "V_GENE": None,
            "V_IDENTITY": None,
            "J_GENE": None,
            "J_IDENTITY": None,
            "NUMBERING": anarci_output[0][0][0][0],
        }

    return output_dict


def get_sequence_annotations(
    sequence: str,
    allow: Optional[set] = {"H", "K", "L"},
    scheme: Optional[str] = "chothia",
    cdr1_scheme={"H": range(26, 35), "L": range(24, 35)},
    cdr2_scheme={"H": range(52, 57), "L": range(50, 57)},
    cdr3_scheme={"H": range(95, 103), "L": range(89, 98)},
    assign_germline: Optional[bool] = True
) -> Dict[str, Dict[str, Union[str, List[str]]]]:

    """
    From a VH or VL amino acid sequences returns the three CDR sequences as determined
    from the input numbering (scheme) and the given ranges.
    default ranges are Chothia CDRs.

    For other numbering schemes see also http://www.bioinf.org.uk/abs/#cdrdef
    Loop    Kabat          AbM    Chothia1    Contact2
    L1    L24--L34    L24--L34    L24--L34    L30--L36
    L2    L50--L56    L50--L56    L50--L56    L46--L55
    L3    L89--L97    L89--L97    L89--L97    L89--L96
    H1    H31--H35B   H26--H35B   H26--H32..34  H30--H35B
    H1    H31--H35    H26--H35    H26--H32    H30--H35
    H2    H50--H65    H50--H58    H52--H56    H47--H58
    H3    H95--H102   H95--H102   H95--H102   H93--H101

    For generic Chothia identification can set auto_detect_chain_type=True and use:
    cdr1_scheme={'H': range(26, 35), 'L': range(24, 35)}
    cdr2_scheme={'H': range(52, 57), 'L': range(50, 57)}
    cdr3_scheme={'H': range(95, 103), 'L': range(89, 98)}

    ============================================================================
    Note:
    * Gracefully stolen and refactored get_cdr_simple() from Parapred source code.
    * In the original scheme arguments did not account range() last value as being non-inclusive. Added +1 to correct this.
    * Returns a nested dictionary with numbering scheme positions for each CDR residue and CDR lengths
    """

    anarci_output = run_and_parse_anarci(sequence, scheme=scheme, allow=allow, assign_germline=assign_germline)
    numbering = anarci_output["NUMBERING"]  # numbering starts with 1 and not 0
    chain_type = anarci_output["CHAIN_TYPE"]

    if chain_type == "K" and chain_type not in cdr1_scheme:
        chain_type = "L"
    if chain_type not in cdr1_scheme:
        raise Exception(f"chain_type {chain_type} is not in input CDR scheme")

    cdr1_scheme = cdr1_scheme[chain_type]
    cdr2_scheme = cdr2_scheme[chain_type]
    cdr3_scheme = cdr3_scheme[chain_type]

    # extract CDR sequences
    cdr1, cdr2, cdr3 = "", "", ""
    cdr1_numbering, cdr2_numbering, cdr3_numbering = [], [], []
    for num_tuple, res in numbering:
        residue_position = str(num_tuple[0]) + num_tuple[1].rstrip()
        if num_tuple[0] in cdr1_scheme:
            cdr1 += res
            if res != "-":
                cdr1_numbering.append(residue_position)
        elif num_tuple[0] in cdr2_scheme:
            cdr2 += res
            if res != "-":
                cdr2_numbering.append(residue_position)
        elif num_tuple[0] in cdr3_scheme:
            cdr3 += res
            if res != "-":
                cdr3_numbering.append(residue_position)

    annotation_dict = {
        "CDR1": cdr1.replace("-", ""),
        "CDR1_NUMBERING": cdr1_numbering,
        "CDR1_LEN": len(cdr1_numbering),
        "CDR2": cdr2.replace("-", ""),
        "CDR2_NUMBERING": cdr2_numbering,
        "CDR2_LEN": len(cdr2_numbering),
        "CDR3": cdr3.replace("-", ""),
        "CDR3_NUMBERING": cdr3_numbering,
        "CDR3_LEN": len(cdr3_numbering),
    }

    annotation_dict = {**annotation_dict, **anarci_output}
    del annotation_dict["NUMBERING"]

    return annotation_dict


def get_annotations(
    sequence: str,
    assign_germline: Optional[bool] = True,
    scheme: Optional[str] = "imgt",
    cdr_scheme: Optional[str] = "north",
    num_extra_residues: Optional[int] = 2,
) -> Dict[str, str]:

    """
    CDRs prediction for a given VH or VL sequence.
    Convenience wrapper around get_sequence_annotations() that includes already defined CDR schemes.
    """

    if cdr_scheme == "north":
        cdr1_scheme = {
            "H": range(24 - num_extra_residues, 41 + num_extra_residues),
            "L": range(24 - num_extra_residues, 41 + num_extra_residues),
        }
        cdr2_scheme = {
            "H": range(55 - num_extra_residues, 67 + num_extra_residues),
            "L": range(55 - num_extra_residues, 70 + num_extra_residues),
        }
        cdr3_scheme = {
            "H": range(105 - num_extra_residues, 118 + num_extra_residues),
            "L": range(105 - num_extra_residues, 118 + num_extra_residues),
        }
    elif cdr_scheme == "chothia":
        cdr1_scheme = {
            "H": range(26 - num_extra_residues, 35 + num_extra_residues),
            "L": range(24 - num_extra_residues, 35 + num_extra_residues),
        }
        cdr2_scheme = {
            "H": range(52 - num_extra_residues, 57 + num_extra_residues),
            "L": range(50 - num_extra_residues, 57 + num_extra_residues),
        }
        cdr3_scheme = {
            "H": range(95 - num_extra_residues, 103 + num_extra_residues),
            "L": range(89 - num_extra_residues, 98 + num_extra_residues),
        }

    annotations = get_sequence_annotations(
        sequence,
        scheme=scheme,
        cdr1_scheme=cdr1_scheme,
        cdr2_scheme=cdr2_scheme,
        cdr3_scheme=cdr3_scheme,
        assign_germline=assign_germline
    )

    return annotations


def annotations_for_df(
    df: pd.DataFrame,
    aa_sequence_col_name: str,
    assign_germline: Optional[bool] = True,
    scheme: Optional[str] = "imgt",
    cdr_scheme: Optional[str] = "north",
    num_extra_residues: Optional[int] = 2,
) -> pd.DataFrame:

    """
    Annotates sequences in pandas dataframe with CDRs and germline genes.
    """

    def assign_annotations(row):

        try:
            annotations = get_annotations(row[aa_sequence_col_name], assign_germline=assign_germline, scheme=scheme)
        except Exception:
            annotations = {
                "CDR1": None,
                "CDR1_NUMBERING": None,
                "CDR1_LEN": None,
                "CDR2": None,
                "CDR2_NUMBERING": None,
                "CDR2_LEN": None,
                "CDR3": None,
                "CDR3_NUMBERING": None,
                "CDR3_LEN": None,
                "CHAIN_TYPE": None,
                "IDENTITY_SPECIES": None,
                "V_GENE": None,
                "V_IDENTITY": None,
                "J_GENE": None,
                "J_IDENTITY": None,
                "NUMBERING": None,
            }
        for key, value in annotations.items():
            row[key] = value
        return row

    df = df.parallel_apply(assign_annotations, axis=1)

    return df


def check_clonotype(
    probe_v_gene: str,
    probe_j_gene: str,
    probe_vh_cdr3_aa_seq: str,
    target_v_gene: str,
    target_j_gene: str,
    target_vh_cdr3_aa_seq: str,
    num_extra_residues: Optional[int] = 0,
) -> float:

    """
    Compares two VH sequences to check if clonotypes match.
    Returns sequence identity of HCDR3 if clonotype is the same i.e. VH and JH genes are the same.
    If clonotypes do not match, sequence identity == 0.
    """

    sequence_identity = 0

    probe_cdrh3_len = len(probe_vh_cdr3_aa_seq)
    if (
        probe_v_gene == target_v_gene
        and probe_j_gene == target_j_gene
        and probe_cdrh3_len == len(target_vh_cdr3_aa_seq)
    ):

        # inverse hamming distance
        match_count = sum(
            res1 == res2
            for res1, res2 in zip(
                probe_vh_cdr3_aa_seq[num_extra_residues : -1 * num_extra_residues],
                target_vh_cdr3_aa_seq[num_extra_residues : -1 * num_extra_residues],
            )
        )

        sequence_identity = match_count / (probe_cdrh3_len - 2 * num_extra_residues)

    return sequence_identity


def cluster_by_clonotype(
    df: pd.DataFrame,
    identity_threshold: float,
    vh_gene_col_name: str,
    jh_gene_col_name: str,
    vh_cdr3_aa_col_name: str,
    num_extra_residues: Optional[int] = 0,
) -> Dict[int, int]:

    """
    Clusters sequences in the dataframe by clonotype using greedy incremental approach.
    Just a PoC - can be optimized further. The key is to know when to stop (I think).
    """

    count = 1
    cluster_dict = {}
    assigned_id_list = []

    for index, vh, jh, cdr3_aa in zip(
        df.index, df[vh_gene_col_name], df[jh_gene_col_name], df[vh_cdr3_aa_col_name]
    ):

        if index in assigned_id_list:
            continue

        members = [
            index2
            for index2, vh2, jh2, cdr3_aa2 in zip(
                df.index,
                df[vh_gene_col_name],
                df[jh_gene_col_name],
                df[vh_cdr3_aa_col_name],
            )
            if check_clonotype(
                vh,
                jh,
                cdr3_aa,
                vh2,
                jh2,
                cdr3_aa2,
                num_extra_residues,
            )
            >= identity_threshold
            and index2 not in assigned_id_list
        ]

        if len(members) > 1:
            cluster_dict[str(count) + f" (seed: {index})"] = members
            assigned_id_list += members
            count += 1

    return cluster_dict


def get_paratope_probabilities(cdrs: Dict[str, str]) -> Dict[str, List[tuple]]:

    """
    Runs Parapred prediction on a set of CDRs.
    Returns probability dictionary for each residue for being part of a paratope.
    Dictionary value is a list of tuples with residue position, residue and probability.
    """

    paratope_probs = {}
    for cdr, cdr_seq in cdrs.items():
        if cdr not in ["CDR1", "CDR2", "CDR3"]:
            continue
        prob = parapred([cdr_seq])
        paratope_probs[cdr] = [
            (pos, residue, prob[0, pos]) for pos, residue in enumerate(cdr_seq)
        ]

    return paratope_probs


def apply_numbering_scheme_positions(
    prob_dict: Dict[str, List[tuple]],
    numbering_dict: Dict[str, List[str]],
) -> Dict[str, List[tuple]]:

    """
    Applies numbering scheme to get_paratope_probabilities() prediciton dictionary
    to enable structurally equivalence when comparing paratopes.
    """

    numbered_prob_dict = {}
    for cdr_name, cdr in prob_dict.items():
        numbered_prob_dict[cdr_name] = []
        for index, res in enumerate(cdr):
            numbered_prob_dict[cdr_name].append(
                (numbering_dict[f"{cdr_name}_NUMBERING"][index], res[1], res[2])
            )

    return numbered_prob_dict


def apply_paratope_prediction_threshold(
    prob_dict: Dict[str, str], paratope_residue_threshold: float
) -> Dict[str, str]:

    """
    Applies paratope residue prediction threshold on a CDR dictionary.
    Returns dictionary of CDRs with non-paratope residues omitted.
    """

    paratope_dict = {}
    for cdr_name, cdr in prob_dict.items():
        paratope_residue_list = [
            residue for residue in cdr if residue[2] > paratope_residue_threshold
        ]
        paratope_dict[cdr_name] = paratope_residue_list

    return paratope_dict


def get_paratope_string(
    paratope_probs: Dict[str, List[tuple]],
    paratope_residue_threshold: Optional[float] = 0.67,
) -> str:

    """
    Returns paratope as a string for a given dictionary of CDRs.
    Non-paratope residues replaced with '-' and CDRs separated by spaces.
    """

    paratope_str = ""
    for cdr_name, cdr in paratope_probs.items():
        for res in cdr:
            if res[2] > paratope_residue_threshold:
                paratope_str += res[1]
            else:
                paratope_str += "-"
        paratope_str += " " * 4

    return paratope_str.rstrip()


def parapred_for_df(
    df: pd.DataFrame, paratope_residue_threshold: Optional[float] = 0.67
) -> pd.DataFrame:

    """
    Runs parapred on CDRs in pandas dataframe.
    """

    def run_parapred(cdr1, cdr2, cdr3, threshold=paratope_residue_threshold):

        try:
            prob_dict = get_paratope_probabilities({"CDR1": cdr1, "CDR2": cdr2, "CDR3": cdr3})
        except Exception:
            prob_dict = None
        return prob_dict

    df["PARATOPE_PROBS"] = df[["CDR1", "CDR2", "CDR3"]].apply(lambda x: run_parapred(*x), axis=1)

    return df


def paratopes_for_df(
    df: pd.DataFrame, paratope_residue_threshold: Optional[float] = 0.67
) -> pd.DataFrame:

    """
    Reformats parapred output in pandas dataframe.
    """

    def reformat_parapred_output(row, threshold=paratope_residue_threshold):

        try:
            prob_dict_numbered = apply_numbering_scheme_positions(
                row['PARATOPE_PROBS'], row[["CDR1_NUMBERING", "CDR2_NUMBERING", "CDR3_NUMBERING"]]
            )
            paratope_dict = apply_paratope_prediction_threshold(
                row['PARATOPE_PROBS'], paratope_residue_threshold
            )
            paratope_dict_numbered = apply_paratope_prediction_threshold(
                prob_dict_numbered, paratope_residue_threshold
            )
            paratope = get_paratope_string(row['PARATOPE_PROBS'], threshold)
        except Exception:
            prob_dict_numbered = None
            paratope_dict = None
            paratope_dict_numbered = None
            paratope = None

        return prob_dict_numbered, paratope_dict, paratope_dict_numbered, paratope

    df["PARATOPE_PROBS_NUMBERED"],  df["PARATOPE_DICT"], df["PARATOPE_DICT_NUMBERED"], df["PARATOPE"] = zip(*df.parallel_apply(reformat_parapred_output, axis=1))

    return df


def check_paratope_positional(
    probe_cdr1_aa_seq: str,
    probe_cdr2_aa_seq: str,
    probe_cdr3_aa_seq: str,
    probe_paratope_dict: Dict[str, Dict[str, str]],
    target_cdr1_aa_seq: str,
    target_cdr2_aa_seq: str,
    target_cdr3_aa_seq: str,
    target_paratope_dict: Dict[str, Dict[str, str]],
    ignore_paratope_length_differences: Optional[bool] = False,
) -> float:

    """
    Compares two sequences to check for paratope similarity using positional equivalence.
    Returns sequence identity between the two paratopes.
    """

    sequence_identity = 0
    if (
        len(probe_cdr1_aa_seq) == len(target_cdr1_aa_seq)
        and len(probe_cdr2_aa_seq) == len(target_cdr2_aa_seq)
        and len(probe_cdr3_aa_seq) == len(target_cdr3_aa_seq)
    ):

        match_count = sum(
            res_num in target_paratope_dict[cdr]
            and res == target_paratope_dict[cdr][res_num]
            for cdr, residue_dict in probe_paratope_dict.items()
            for res_num, res in residue_dict.items()
        )

        probe_paratope_len = sum(
            len(value) for key, value in probe_paratope_dict.items()
        )
        target_paratope_len = sum(
            len(value) for key, value in target_paratope_dict.items()
        )
        sequence_identity = match_count / min(probe_paratope_len, target_paratope_len)

    return sequence_identity


def check_paratope_structural(
    probe_cdr1_aa_seq: str,
    probe_cdr2_aa_seq: str,
    probe_cdr3_aa_seq: str,
    probe_paratope_dict: Dict[str, Dict[str, str]],
    target_cdr1_aa_seq: str,
    target_cdr2_aa_seq: str,
    target_cdr3_aa_seq: str,
    target_paratope_dict: Dict[str, Dict[str, str]],
    ignore_paratope_length_differences: Optional[bool] = False,
) -> float:

    """
    Compares two sequences to check for paratope similarity using IMGT structural equivalence.
    Returns sequence identity between the two paratopes.
    """

    residue_type_dict = {
        "G": "small",
        "A": "small",
        "S": "nucleophilic",
        "T": "nucleophilic",
        "C": "nucleophilic",
        "V": "hydrophobic",
        "L": "hydrophobic",
        "I": "hydrophobic",
        "M": "hydrophobic",
        "P": "hydrophobic",
        "F": "aromatic",
        "W": "aromatic",
        "Y": "aromatic",
        "D": "acidic",
        "E": "acidic",
        "N": "amine",
        "Q": "amine",
        "K": "basic",
        "H": "basic",
        "R": "basic",
    }

    # counting for residue matches in the paratope
    match_count = 0
    for cdr, residue_dict in probe_paratope_dict.items():
        for res_num, res in residue_dict.items():
            if res_num in target_paratope_dict[cdr]:
                if res == target_paratope_dict[cdr][res_num]:
                    match_count += 1
                elif (
                    residue_type_dict[res]
                    == residue_type_dict[target_paratope_dict[cdr][res_num]]
                ):
                    match_count += 0.5

    probe_paratope_len = sum(len(value) for key, value in probe_paratope_dict.items())
    target_paratope_len = sum(len(value) for key, value in target_paratope_dict.items())
    if ignore_paratope_length_differences:
        sequence_identity = match_count / min(probe_paratope_len, target_paratope_len)
    else:
        sequence_identity = match_count / max(probe_paratope_len, target_paratope_len)

    return sequence_identity


def cluster_by_paratope(
    df: pd.DataFrame,
    cdr1_aa_seq_col_name: str,
    cdr2_aa_seq_col_name: str,
    cdr3_aa_seq_col_name: str,
    paratope_dict_col_name: str,
    identity_threshold: float,
    ignore_cdr_lengths: Optional[bool] = False,
    ignore_paratope_length_differences: Optional[bool] = False,
) -> Dict[int, int]:

    """
    Clusters sequences in the dataframe by paratope using greedy incremental approach.

    ignore_cdr_lenghts = False, matches sequences only with the same CDR lengths.
    Residue postions are used as provided by Parapred

    ignore_cdr_lenghts = True, uses structural equivalence provided by IMGT numbering
    and compares different lenght CDRs. Also assigns scores for similar residues when calculating sequence identity.

    ignore_paratope_length_differences = False, divides the the number of paratope
    matches by the longer paratope (versus shorter paratope as in the to the original implementation).
    More sensitive to paratope residue count mismatches.
    """

    # setting up paratope comparison method
    if ignore_cdr_lengths is True:
        check_paratope = check_paratope_structural
    else:
        check_paratope = check_paratope_positional

    count = 1
    cluster_dict = {}
    assigned_id_list = []

    for index, cdr1, cdr2, cdr3, paratope in zip(
        df.index,
        df[cdr1_aa_seq_col_name],
        df[cdr2_aa_seq_col_name],
        df[cdr3_aa_seq_col_name],
        df[paratope_dict_col_name],
    ):

        if index in assigned_id_list:
            continue

        members = [
            index2
            for index2, cdr1_2, cdr2_2, cdr3_2, paratope2 in zip(
                df.index,
                df[cdr1_aa_seq_col_name],
                df[cdr2_aa_seq_col_name],
                df[cdr3_aa_seq_col_name],
                df[paratope_dict_col_name],
            )
            if check_paratope(
                cdr1,
                cdr2,
                cdr3,
                paratope,
                cdr1_2,
                cdr2_2,
                cdr3_2,
                paratope2,
                ignore_paratope_length_differences,
            )
            >= identity_threshold
            and index2 not in assigned_id_list
        ]

        if len(members) > 1:
            cluster_dict[str(count) + f" (seed: {index})"] = members
            assigned_id_list += members
            count += 1

    return cluster_dict


def cluster(
    df: pd.DataFrame,
    vh_aa_sequence_col_name: str,
    scheme: Optional[str] = "imgt",
    cdr_scheme: Optional[str] = "north",
    num_extra_residues: Optional[int] = 2,
    paratope_residue_threshold: Optional[float] = 0.67,
    paratope_identity_threshold: Optional[float] = 0.75,
    clonotype_identity_threshold: Optional[float] = 0.72,
    structural_equivalence: Optional[bool] = True,
    ignore_paratope_length_differences: Optional[bool] = False,
    perform_clonotyping: Optional[bool] = True,
) -> pd.DataFrame:

    """
    Annotates and clusters VH sequences by clonotype and paratope in the pandas dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe to cluster.
    vh_aa_sequence_col_name : str
        column name for VH sequences.
    scheme : str, default "imgt"
        numbering scheme to use.
    cdr_scheme : str, default "north"
        CDR definition to use. Only North and Chothia supported.
    num_extra_residues : int, default 2
        include extra residues at the start and end of each CDR.
    paratope_residue_threshold : float, default 0.67
        paratope residue call threshold for parapred predictions.
    paratope_identity_threshold : float, default 0.75
        paratope sequence identity value at which paratope are considered the same.
    clonotype_identity_threshold : float, default 0.72
        clonotype sequence identity value at which clonotype are considered the same.
    structural_equivalence : bool, default True
        specify whether positional or structural equivalence as assigned by IMGT should be used.
    ignore_paratope_length_differences : bool, default True
        specify whether paratope length mismatches should be taken into an account when calcualting similarity between paratopes.
        Only applicaple if structural_equivalence=True
    perform_clonotyping : bool, default True
        specify if clonotyping should be performed.

    Returns
    -------
    pd.DataFrame
    """

    print(figlet_format("PaCPaC", font="banner"))

    # slicing rows with NaN for sequences
    nan_df = df[df[vh_aa_sequence_col_name].isnull()]
    df = df[df[vh_aa_sequence_col_name].notnull()]
    if not nan_df.empty:
        print("Not to worry, we are still flying half a dataset")

    print("I'll try annotating, that's a good trick")
    df = annotations_for_df(
        df,
        vh_aa_sequence_col_name,
        assign_germline=perform_clonotyping,
        scheme=scheme,
        cdr_scheme=cdr_scheme,
        num_extra_residues=num_extra_residues,
    )

    # exclude sequences where anarci has failed
    nan_df2 = df[df["CDR3"].isnull()]
    df = df[df["CDR3"].notnull()]

    df.sort_values(
        ["CDR3_LEN", "CDR2_LEN", "CDR1_LEN", vh_aa_sequence_col_name],
        ascending=False,
        inplace=True,
    )

    def assign_cluster(row, cluster_dict):
        for cluster_no, sequences in cluster_dict.items():
            if row.name in sequences:
                return cluster_no

    if perform_clonotyping is True:
        print("Now THIS is clonotype clustering")
        clonotype_cluster_dict = cluster_by_clonotype(
            df,
            clonotype_identity_threshold,
            "V_GENE",
            "J_GENE",
            "CDR3",
            num_extra_residues=num_extra_residues,
        )
        df["CLONOTYPE_CLUSTER"] = df.apply(
            assign_cluster, args=(clonotype_cluster_dict,), axis=1
        )
        print("Your clonotypes are very impressive. You must be very proud")

    print("Learning to stop worrying and falling in love with the paratope")
    df = parapred_for_df(df, paratope_residue_threshold=paratope_residue_threshold)
    df = paratopes_for_df(df, paratope_residue_threshold=paratope_residue_threshold)

    # exclude sequences where parapred has failed
    nan_df3 = df[df["PARATOPE"].isnull()]
    df = df[df["PARATOPE"].notnull()]

    print("Hold on. This whole paratope clustering was your idea")
    paratope_dict_col = "PARATOPE_DICT"
    if structural_equivalence is True:
        paratope_dict_col = "PARATOPE_DICT_NUMBERED"

    # reformatting paratope dict column
    df["PARATOPE_DICT_REFORMAT"] = [
        {
            cdr: {residue[0]: residue[1] for residue in value}
            for cdr, value in paratope_dict.items()
        }
        for paratope_dict in df[paratope_dict_col]
    ]

    paratope_cluster_dict = cluster_by_paratope(
        df,
        "CDR1",
        "CDR2",
        "CDR3",
        "PARATOPE_DICT_REFORMAT",
        paratope_identity_threshold,
        ignore_cdr_lengths=structural_equivalence,
        ignore_paratope_length_differences=ignore_paratope_length_differences,
    )
    df["PARATOPE_CLUSTER"] = df.apply(
        assign_cluster, args=(paratope_cluster_dict,), axis=1
    )
    print("200,000 paratopes are ready, with a million more well on the way")

    df = pd.concat(
        [df, nan_df, nan_df2, nan_df3], ignore_index=False, axis=0, sort=False
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
        ],
        axis=1,
        inplace=True,
    )
    df.sort_index(inplace=True)

    print("Another happy clustering")

    return df


def probe(
    probe_sequence: str,
    df: pd.DataFrame,
    vh_aa_sequence_col_name: str,
    scheme: Optional[str] = "imgt",
    cdr_scheme: Optional[str] = "north",
    num_extra_residues: Optional[int] = 2,
    paratope_residue_threshold: Optional[float] = 0.67,
    paratope_identity_threshold: Optional[float] = 0.75,
    clonotype_identity_threshold: Optional[float] = 0.72,
    structural_equivalence: Optional[bool] = True,
    ignore_paratope_length_differences: Optional[bool] = False,
    perform_clonotyping: Optional[bool] = True,
) -> pd.DataFrame:

    """
    Probe sequences in the pandas dataframe for similar paratopes and clonotypes.

    Parameters
    ----------
    probe_sequence : str
        VH sequence to use as a probe.
    df : pd.DataFrame
        pandas dataframe to cluster.
    vh_aa_sequence_col_name : str
        column name for VH sequences.
    scheme : str, default "imgt"
        numbering scheme to use.
    cdr_scheme : str, default "north"
        CDR definition to use. Only North and Chothia supported.
    num_extra_residues : int, default 2
        include extra residues at the start and end of each CDR.
    paratope_residue_threshold : float, default 0.67
        paratope residue call threshold for parapred predictions.
    paratope_identity_threshold : float, default 0.75
        paratope sequence identity value at which paratope are considered the same.
    clonotype_identity_threshold : float, default 0.72
        clonotype sequence identity value at which clonotype are considered the same.
    structural_equivalence : bool, default True
        specify whether positional or structural equivalence as assigned by IMGT should be used.
    ignore_paratope_length_differences : bool, default True
        specify whether paratope length mismatches should be taken into an account when calcualting similarity between paratopes.
        Only applicaple if structural_equivalence=True
    perform_clonotyping : bool, default True
        specify if clonotyping should be performed.

    Returns
    -------
    pd.DataFrame
    """

    print(figlet_format("PaCPaC", font="banner"))

    # slicing rows with NaN for sequences
    nan_df = df[df[vh_aa_sequence_col_name].isnull()]
    df = df[df[vh_aa_sequence_col_name].notnull()]
    if not nan_df.empty:
        print("Not to worry, we are still flying half a dataset")

    # create paratope for the probe
    annotations = get_annotations(
        probe_sequence,
        assign_germline=perform_clonotyping,
        scheme=scheme,
        cdr_scheme=cdr_scheme,
        num_extra_residues=num_extra_residues,
    )
    prob_dict = get_paratope_probabilities(annotations)

    paratope_dict_col = "PARATOPE_DICT"
    if structural_equivalence is True:
        paratope_dict_col = "PARATOPE_DICT_NUMBERED"
        prob_dict = apply_numbering_scheme_positions(prob_dict, annotations)

    paratope = apply_paratope_prediction_threshold(
        prob_dict, paratope_residue_threshold
    )
    annotations[paratope_dict_col] = paratope
    annotations["PARATOPE_DICT_REFORMAT"] = {
        cdr: {residue[0]: residue[1] for residue in value}
        for cdr, value in annotations[paratope_dict_col].items()
    }
    probe_dict = annotations

    # annotate sequences with genes and cdrs
    print("I'll try annotating, that's a good trick")
    df = annotations_for_df(
        df,
        vh_aa_sequence_col_name,
        scheme=scheme,
        cdr_scheme=cdr_scheme,
        num_extra_residues=num_extra_residues,
    )

    # exclude sequences where anarci has failed
    nan_df2 = df[df["CDR3"].isnull()]
    df = df[df["CDR3"].notnull()]

    # create paratopes for sequences in df
    print("Learning to stop worrying and falling in love with the paratope")
    df = parapred_for_df(df, paratope_residue_threshold=paratope_residue_threshold)
    df = paratopes_for_df(df, paratope_residue_threshold=paratope_residue_threshold)

    # exclude sequences where parapred has failed
    nan_df3 = df[df["PARATOPE"].isnull()]
    df = df[df["PARATOPE"].notnull()]

    # reformatting paratope dict column
    df["PARATOPE_DICT_REFORMAT"] = [
        {
            cdr: {residue[0]: residue[1] for residue in value}
            for cdr, value in paratope_dict.items()
        }
        for paratope_dict in df[paratope_dict_col]
    ]

    # setting up paratope comparison method
    if structural_equivalence is True:
        check_paratope = check_paratope_structural
    else:
        check_paratope = check_paratope_positional

    # probing with paratope
    print("This is where the paratope probing begins")
    df["PARATOPE_MATCH"] = [
        True
        if check_paratope(
            probe_dict["CDR1"],
            probe_dict["CDR2"],
            probe_dict["CDR3"],
            probe_dict["PARATOPE_DICT_REFORMAT"],
            cdr1,
            cdr2,
            cdr3,
            paratope,
            ignore_paratope_length_differences=ignore_paratope_length_differences,
        )
        >= paratope_identity_threshold
        else False
        for cdr1, cdr2, cdr3, paratope in zip(
            df["CDR1"], df["CDR2"], df["CDR3"], df["PARATOPE_DICT_REFORMAT"]
        )
    ]

    # probing with clonotype
    if perform_clonotyping is True:
        print("We're just clonotypes, sir. We're meant to be expendable")
        df["CLONOTYPE_MATCH"] = [
            True
            if check_clonotype(
                probe_dict["V_GENE"],
                probe_dict["J_GENE"],
                probe_dict["CDR3"],
                vh,
                jh,
                cdr3_aa,
                num_extra_residues=num_extra_residues,
            )
            >= clonotype_identity_threshold
            else False
            for vh, jh, cdr3_aa in zip(df["V_GENE"], df["J_GENE"], df["CDR3"])
        ]
    else:
        df["CLONOTYPE_MATCH"] = None

    df["PREDICTION_SPACE"] = [
        "both"
        if p_match is True and c_match is True
        else "paratope-only"
        if p_match is True
        else "clonotype-only"
        if c_match is True
        else None
        for p_match, c_match in zip(df["PARATOPE_MATCH"], df["CLONOTYPE_MATCH"])
    ]

    df = pd.concat(
        [df, nan_df, nan_df2, nan_df3], ignore_index=False, axis=0, sort=False
    )
    df.drop(
        [
            "CDR1_NUMBERING",
            "CDR2_NUMBERING",
            "CDR3_NUMBERING",
            "PARATOPE_MATCH",
            "CLONOTYPE_MATCH",
            "PARATOPE_PROBS",
            "PARATOPE_PROBS_NUMBERED",
            "PARATOPE_DICT",
            "PARATOPE_DICT_NUMBERED",
            "PARATOPE_DICT_REFORMAT",
        ],
        axis=1,
        inplace=True,
    )

    print("Another happy probing")

    return df
