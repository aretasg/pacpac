# Author: Aretas Gaspariunas

from typing import List, Dict, Optional, Iterable, Union, Callable

import pandas as pd
from pandarallel import pandarallel
from anarci import anarci
from pyfiglet import figlet_format
import numpy as np
from numba import njit, types
from numba.typed import Dict as numbaDict

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
    assign_germline: Optional[bool] = True,
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
    cdr1_scheme: Optional[Dict[str, Iterable]] = {
        "H": range(26, 33),
        "L": range(24, 35),
    },
    cdr2_scheme: Optional[Dict[str, Iterable]] = {
        "H": range(52, 57),
        "L": range(50, 57),
    },
    cdr3_scheme: Optional[Dict[str, Iterable]] = {
        "H": range(95, 103),
        "L": range(89, 98),
    },
    assign_germline: Optional[bool] = True,
) -> Dict[str, Union[str, int, List[str]]]:

    """
    From a VH or VL amino acid sequences returns the three CDR sequences as determined
    from the input numbering (scheme) and the given ranges.
    default ranges are Chothia CDRs.

    ============================================================================
    Note:
    * Gracefully stolen and refactored get_cdr_simple() from Parapred source code.
    * Returns a dictionary with CDR sequences, numbering scheme positions for each CDR residue and CDR lengths
    """

    anarci_output = run_and_parse_anarci(
        sequence, scheme=scheme, allow=allow, assign_germline=assign_germline
    )
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
            if res != "-":
                cdr1_numbering.append(residue_position)
                cdr1 += res
        elif num_tuple[0] in cdr2_scheme:
            if res != "-":
                cdr2_numbering.append(residue_position)
                cdr2 += res
        elif num_tuple[0] in cdr3_scheme:
            if res != "-":
                cdr3_numbering.append(residue_position)
                cdr3 += res

    annotation_dict = {
        "CDR1": cdr1,
        "CDR1_NUMBERING": cdr1_numbering,
        "CDR2": cdr2,
        "CDR2_NUMBERING": cdr2_numbering,
        "CDR3": cdr3,
        "CDR3_NUMBERING": cdr3_numbering,
    }

    annotation_dict = {**annotation_dict, **anarci_output}
    del annotation_dict["NUMBERING"]

    return annotation_dict


def get_annotations(
    sequence: str,
    assign_germline: Optional[bool] = True,
    scheme: Optional[str] = "chothia",
    cdr_scheme: Optional[str] = "chothia",
    num_extra_residues: Optional[int] = 2,
) -> Dict[str, str]:

    """
    Annotation and CDRs definition for a given VH or VL sequence.
    Convenience wrapper around get_sequence_annotations() with defined CDR schemas.
    """

    if cdr_scheme in ("imgt"):
        scheme = "imgt"
    elif cdr_scheme in ("chothia", "contact") and scheme not in ("chothia", "martin"):
        scheme = "chothia"

    if cdr_scheme == "chothia":
        cdr1_scheme = {
            "H": range(26 - num_extra_residues, 33 + num_extra_residues),
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
    elif cdr_scheme == "imgt":
        cdr1_scheme = {
            "H": range(27 - num_extra_residues, 39 + num_extra_residues),
            "L": range(27 - num_extra_residues, 39 + num_extra_residues),
        }
        cdr2_scheme = {
            "H": range(56 - num_extra_residues, 66 + num_extra_residues),
            "L": range(56 - num_extra_residues, 66 + num_extra_residues),
        }
        cdr3_scheme = {
            "H": range(105 - num_extra_residues, 118 + num_extra_residues),
            "L": range(105 - num_extra_residues, 118 + num_extra_residues),
        }
    elif cdr_scheme == "north" and scheme == "imgt":
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
    elif cdr_scheme == "north" and scheme != "imgt":
        cdr1_scheme = {
            "H": range(21 - num_extra_residues, 38 + num_extra_residues),
            "L": range(22 - num_extra_residues, 37 + num_extra_residues),
        }
        cdr2_scheme = {
            "H": range(49 - num_extra_residues, 61 + num_extra_residues),
            "L": range(47 - num_extra_residues, 59 + num_extra_residues),
        }
        cdr3_scheme = {
            "H": range(91 - num_extra_residues, 105 + num_extra_residues),
            "L": range(87 - num_extra_residues, 102 + num_extra_residues),
        }
    elif cdr_scheme == "contact":
        cdr1_scheme = {
            "H": range(30 - num_extra_residues, 36 + num_extra_residues),
            "L": range(30 - num_extra_residues, 37 + num_extra_residues),
        }
        cdr2_scheme = {
            "H": range(47 - num_extra_residues, 59 + num_extra_residues),
            "L": range(46 - num_extra_residues, 56 + num_extra_residues),
        }
        cdr3_scheme = {
            "H": range(93 - num_extra_residues, 102 + num_extra_residues),
            "L": range(89 - num_extra_residues, 97 + num_extra_residues),
        }

    annotations = get_sequence_annotations(
        sequence,
        scheme=scheme,
        cdr1_scheme=cdr1_scheme,
        cdr2_scheme=cdr2_scheme,
        cdr3_scheme=cdr3_scheme,
        assign_germline=assign_germline,
    )

    return annotations


def annotations_for_df(
    df: pd.DataFrame,
    aa_sequence_col_name: str,
    assign_germline: Optional[bool] = True,
    scheme: Optional[str] = "chothia",
    cdr_scheme: Optional[str] = "chothia",
    num_extra_residues: Optional[int] = 2,
) -> pd.DataFrame:

    """
    Annotates sequences in pandas dataframe with CDRs and germline genes.
    """

    def assign_annotations(row):

        try:
            annotations = get_annotations(
                row[aa_sequence_col_name],
                assign_germline=assign_germline,
                scheme=scheme,
            )
        except Exception:
            annotations = {
                "CDR1": None,
                "CDR1_NUMBERING": None,
                "CDR2": None,
                "CDR2_NUMBERING": None,
                "CDR3": None,
                "CDR3_NUMBERING": None,
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
    start_cdr: int,
    end_cdr: int,
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
                for res1, res2 in zip(
                    probe_cdr3_aa_seq[start_cdr:end_cdr],
                    target_cdr3_aa_seq[start_cdr:end_cdr],
                )
                if res1 == res2
            ]
        )

        sequence_identity = match_count / (probe_cdr3_len - 2 * start_cdr)

    return sequence_identity


@njit(cache=True)
def cluster_by_clonotype(
    index_list: List[int],
    vh_gene_list: List[str],
    jh_gene_list: List[str],
    vh_cdr3_len_list: List[int],
    vh_cdr3_aa_list: List[str],
    identity_threshold: float,
    num_extra_residues: Optional[int] = 0,
) -> Dict[tuple, List[int]]:

    """
    Clusters sequences in the dataframe by clonotype using greedy incremental approach.
    Just a PoC - can be optimized further. The key is to know when to stop (I think).
    """

    count = 1
    cluster_dict = dict()
    k = {(1, 2): np.arange(1), (3, 4): np.arange(2)}  # dict types for numba compiler
    assigned_id_list = [-1]  # declaring a list of ints for numba compiler

    end_cdr = -1 * num_extra_residues
    if num_extra_residues == 0:
        end_cdr = None

    for index, cdr3_len, vh, jh, cdr3_aa in zip(
        index_list, vh_cdr3_len_list, vh_gene_list, jh_gene_list, vh_cdr3_aa_list
    ):

        if index in set(assigned_id_list):
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
                num_extra_residues,
                end_cdr,
            )
            >= identity_threshold
            and index2 not in set(assigned_id_list)
        ]

        assigned_id_list += members
        if len(members) > 1:
            cluster_dict[(count, index)] = np.array(members)
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
            prob_dict = get_paratope_probabilities(
                {"CDR1": cdr1, "CDR2": cdr2, "CDR3": cdr3}
            )
        except Exception:
            prob_dict = None
        return prob_dict

    df["PARATOPE_PROBS"] = df[["CDR1", "CDR2", "CDR3"]].apply(
        lambda x: run_parapred(*x), axis=1
    )

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
                row["PARATOPE_PROBS"],
                row[["CDR1_NUMBERING", "CDR2_NUMBERING", "CDR3_NUMBERING"]],
            )
            paratope_dict = apply_paratope_prediction_threshold(
                row["PARATOPE_PROBS"], paratope_residue_threshold
            )
            paratope_dict_numbered = apply_paratope_prediction_threshold(
                prob_dict_numbered, paratope_residue_threshold
            )
            paratope = get_paratope_string(row["PARATOPE_PROBS"], threshold)
        except Exception:
            prob_dict_numbered = None
            paratope_dict = None
            paratope_dict_numbered = None
            paratope = None

        return prob_dict_numbered, paratope_dict, paratope_dict_numbered, paratope

    (
        df["PARATOPE_PROBS_NUMBERED"],
        df["PARATOPE_DICT"],
        df["PARATOPE_DICT_NUMBERED"],
        df["PARATOPE"],
    ) = zip(*df.parallel_apply(reformat_parapred_output, axis=1))

    return df


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


@njit(cache=True)
def check_paratope_structural(
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
def cluster_by_paratope(
    index_list: List[int],
    cdr1_aa_seq_list: List[str],
    cdr2_aa_seq_list: List[str],
    cdr3_aa_seq_list: List[str],
    paratope_len_list: List[int],
    paratope_dict_list: List[Dict[str, Dict[str, str]]],
    identity_threshold: float,
    check_paratope: Optional[Callable] = check_paratope_structural,
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
            if check_paratope(
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
            and index2 not in set(assigned_id_list)
        ]

        assigned_id_list += members
        if len(members) > 1:
            cluster_dict[(count, index)] = np.array(members)
            count += 1

    return cluster_dict


def convert_to_typed_numba_dict(
    input_dict: Dict[str, Dict[str, str]]
) -> Dict[str, Dict[str, str]]:

    # https://github.com/numba/numba/issues/6191#issuecomment-684022879

    inner_dict_type = types.DictType(types.unicode_type, types.unicode_type)
    d = numbaDict.empty(
        key_type=types.unicode_type,
        value_type=inner_dict_type,
    )

    for cdr, res_dict in input_dict.items():

        inner_d = numbaDict.empty(
            key_type=types.unicode_type,
            value_type=types.unicode_type,
        )

        inner_d.update(res_dict)
        d[cdr] = inner_d

    return d


def cluster(
    df: pd.DataFrame,
    vh_aa_sequence_col_name: str,
    scheme: Optional[str] = "chothia",
    cdr_scheme: Optional[str] = "chothia",
    num_extra_residues: Optional[int] = 2,
    paratope_residue_threshold: Optional[float] = 0.67,
    paratope_identity_threshold: Optional[float] = 0.75,
    clonotype_identity_threshold: Optional[float] = 0.72,
    structural_equivalence: Optional[bool] = True,
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
    scheme : str, default "chothia"
        numbering scheme to use. IMGT, Chothia, Martin only.
    cdr_scheme : str, default "chothia"
        CDR definition to use. IMGT, North, Chothia and Contact supported.
        IMGT, North can be used with IMGT numbering scheme.
        North, Chothia, Contact can be used with Chothia, Martin numbering schemes.
        CDR definition with not supported numbering scheme will default to to supported numbering scheme.
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
            choice should be used and paratopes with different CDR lengths should be compared.
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

    df["CDR1_LEN"] = df["CDR1"].astype(str).map(len)
    df["CDR2_LEN"] = df["CDR2"].astype(str).map(len)
    df["CDR3_LEN"] = df["CDR3"].astype(str).map(len)

    df.sort_values(
        ["CDR3_LEN", "CDR2_LEN", "CDR1_LEN", vh_aa_sequence_col_name],
        ascending=False,
        inplace=True,
    )

    def assign_cluster(row, cluster_dict):
        for cluster_no, sequences in cluster_dict.items():
            if row.name in sequences:
                return str(cluster_no[0]) + f" (seed: {cluster_no[1]})"

    if perform_clonotyping is True:

        print("Now THIS is clonotype clustering")
        clonotype_cluster_dict = cluster_by_clonotype(
            df.index.tolist(),
            df["V_GENE"].tolist(),
            df["J_GENE"].tolist(),
            df["CDR3_LEN"].tolist(),
            df["CDR3"].tolist(),
            clonotype_identity_threshold,
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
            cdr: {str(residue[0]): residue[1] for residue in value}
            for cdr, value in paratope_dict.items()
        }
        for paratope_dict in df[paratope_dict_col]
    ]
    df["PARATOPE_DICT_REFORMAT"] = [
        convert_to_typed_numba_dict(paratope_dict)
        for paratope_dict in df["PARATOPE_DICT_REFORMAT"]
    ]
    df["PARATOPE_LEN"] = [
        sum(len(res_dict) for cdr, res_dict in paratope_dict.items())
        for paratope_dict in df["PARATOPE_DICT_REFORMAT"]
    ]

    df.sort_values(
        ["PARATOPE_LEN", vh_aa_sequence_col_name],
        ascending=False,
        inplace=True,
    )

    paratope_cluster_dict = cluster_by_paratope(
        df.index.tolist(),
        df["CDR1"].tolist(),
        df["CDR2"].tolist(),
        df["CDR3"].tolist(),
        df["PARATOPE_LEN"].tolist(),
        df["PARATOPE_DICT_REFORMAT"].tolist(),
        paratope_identity_threshold,
        check_paratope={
            False: check_paratope_equal_len_cdrs,
            True: check_paratope_structural,
        }[structural_equivalence],
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
    scheme: Optional[str] = "chothia",
    cdr_scheme: Optional[str] = "chothia",
    num_extra_residues: Optional[int] = 2,
    paratope_residue_threshold: Optional[float] = 0.67,
    paratope_identity_threshold: Optional[float] = 0.75,
    clonotype_identity_threshold: Optional[float] = 0.72,
    structural_equivalence: Optional[bool] = True,
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
    scheme : str, default "chothia"
        numbering scheme to use.
    cdr_scheme : str, default "chothia"
        CDR definition to use. IMGT, North, Chothia and Contact supported.
        IMGT, North can be used with IMGT numbering scheme.
        North, Chothia, Contact can be used with Chothia, Martin numbering schemes.
        CDR definition with not supported numbering scheme will default to to supported numbering scheme.
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
            choice should be used and paratopes with different CDR lengths should be compared.
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
        cdr: {str(residue[0]): residue[1] for residue in value}
        for cdr, value in annotations[paratope_dict_col].items()
    }
    annotations["PARATOPE_DICT_REFORMAT"] = convert_to_typed_numba_dict(
        annotations["PARATOPE_DICT_REFORMAT"]
    )
    annotations["PARATOPE_LEN"] = sum(
        len(res_dict) for cdr, res_dict in annotations["PARATOPE_DICT_REFORMAT"].items()
    )
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

    df["CDR1_LEN"] = df["CDR1"].astype(str).map(len)
    df["CDR2_LEN"] = df["CDR2"].astype(str).map(len)
    df["CDR3_LEN"] = df["CDR3"].astype(str).map(len)

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
            cdr: {str(residue[0]): residue[1] for residue in value}
            for cdr, value in paratope_dict.items()
        }
        for paratope_dict in df[paratope_dict_col]
    ]
    df["PARATOPE_DICT_REFORMAT"] = [
        convert_to_typed_numba_dict(paratope_dict)
        for paratope_dict in df["PARATOPE_DICT_REFORMAT"]
    ]
    df["PARATOPE_LEN"] = [
        sum(len(res_dict) for cdr, res_dict in paratope_dict.items())
        for paratope_dict in df["PARATOPE_DICT_REFORMAT"]
    ]

    df.sort_values(
        ["PARATOPE_LEN", vh_aa_sequence_col_name],
        ascending=False,
        inplace=True,
    )

    # setting up paratope comparison method
    if structural_equivalence is True:
        check_paratope = check_paratope_structural
    else:
        check_paratope = check_paratope_equal_len_cdrs

    # probing with paratope
    print("This is where the paratope probing begins")
    df["PARATOPE_MATCH"] = [
        True
        if check_paratope(
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
        >= paratope_identity_threshold
        else False
        for cdr1, cdr2, cdr3, paratope_len, paratope in zip(
            df["CDR1"],
            df["CDR2"],
            df["CDR3"],
            df["PARATOPE_LEN"],
            df["PARATOPE_DICT_REFORMAT"],
        )
    ]

    # probing with clonotype
    if perform_clonotyping is True:
        print("We're just clonotypes, sir. We're meant to be expendable")

        end_cdr = -1 * num_extra_residues
        if num_extra_residues == 0:
            end_cdr = None

        df["CLONOTYPE_MATCH"] = [
            True
            if check_clonotype(
                probe_dict["V_GENE"],
                probe_dict["J_GENE"],
                len(probe_dict["CDR3"]),
                probe_dict["CDR3"],
                vh,
                jh,
                cdr3_len,
                cdr3_aa,
                num_extra_residues,
                end_cdr,
            )
            >= clonotype_identity_threshold
            else False
            for vh, jh, cdr3_len, cdr3_aa in zip(
                df["V_GENE"], df["J_GENE"], df["CDR3_LEN"], df["CDR3"]
            )
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
