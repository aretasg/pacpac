# Author: Aretas Gaspariunas

from typing import List, Dict, Optional, Iterable, Union, Tuple, Any
import warnings
import os
from contextlib import redirect_stderr

from anarci import anarci
import pandas as pd
from pandarallel import pandarallel
with redirect_stderr(open(os.devnull, "w")): # disable Keras messages
    import keras

from pacpac.parapred.parapred import predict_sequence_probabilities as parapred
from pacpac.utils import convert_to_typed_numba_dict, rename_dict_keys


# disable TF messages
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# init pandarallel
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
    For VH or VL amino acid sequence returns the three CDR sequences as determined
    from the input numbering (scheme) and the given ranges.
    default ranges are Chothia CDRs.

    ============================================================================
    Note:
    * Gracefully stolen and refactored get_cdr_simple() from Parapred source code.
    * Returns a dictionary with CDR sequences, numbering scheme positions for each CDR residue.
    """

    anarci_output = run_and_parse_anarci(
        sequence, scheme=scheme, allow=allow, assign_germline=assign_germline
    )
    numbering = anarci_output["NUMBERING"]  # numbering starts with 1 and not 0
    chain_type = anarci_output["CHAIN_TYPE"]

    if chain_type == "K" and chain_type not in cdr1_scheme:
        chain_type = "L"
    if chain_type not in cdr1_scheme:
        raise ValueError(f"chain_type {chain_type} is not in input CDR scheme")

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
    Annotation and CDR definition for a given VH or VL sequence.
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
                cdr_scheme=cdr_scheme,
            )
        except TypeError:
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
        except ValueError:
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

        if row["PARATOPE_PROBS"] is not None:
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
        else:
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


def get_residue_token_dict() -> Dict[str, str]:

    # as described by Wong et al., 2020
    # S =  small; N = nucleophilic; H = hydrophobic; A = aromatic; C = acidic; M = amine; B = basic
    residue_token_dict = {
        "G": "S",
        "A": "S",
        "S": "N",
        "T": "N",
        "C": "N",
        "V": "H",
        "L": "H",
        "I": "H",
        "M": "H",
        "P": "H",
        "F": "A",
        "W": "A",
        "Y": "A",
        "D": "C",
        "E": "C",
        "N": "M",
        "Q": "M",
        "K": "B",
        "H": "B",
        "R": "B",
    }

    return residue_token_dict


def annotate_sequence(
    sequence: str,
    scheme: Optional[str] = "chothia",
    cdr_scheme: Optional[str] = "chothia",
    assign_germline: Optional[bool] = True,
    num_extra_residues: Optional[int] = 2,
    paratope_residue_threshold: Optional[float] = 0.67,
    structural_equivalence: Optional[bool] = True,
    tokenize: Optional[bool] = False,
) -> Dict[str, Any]:

    """
    Annotates input VH or VL sequence with anarci and parapred
    """

    annotations = get_annotations(
        sequence,
        assign_germline=assign_germline,
        scheme=scheme,
        cdr_scheme=cdr_scheme,
        num_extra_residues=num_extra_residues,
    )
    prob_dict = get_paratope_probabilities(annotations)

    if structural_equivalence is True:
        prob_dict = apply_numbering_scheme_positions(prob_dict, annotations)

    paratope = apply_paratope_prediction_threshold(
        prob_dict, paratope_residue_threshold
    )
    annotations["PARATOPE_DICT"] = paratope
    if tokenize:
        residue_token_dict = get_residue_token_dict()
        annotations["PARATOPE_DICT_REFORMAT"] = {
            cdr: {str(residue[0]): residue_token_dict[residue[1]] for residue in value}
            for cdr, value in annotations["PARATOPE_DICT"].items()
        }
    else:
        annotations["PARATOPE_DICT_REFORMAT"] = {
            cdr: {str(residue[0]): residue[1] for residue in value}
            for cdr, value in annotations["PARATOPE_DICT"].items()
        }
    annotations["PARATOPE_LEN"] = sum(
        len(res_dict) for cdr, res_dict in annotations["PARATOPE_DICT_REFORMAT"].items()
    )
    annotations["PARATOPE_DICT_REFORMAT"] = convert_to_typed_numba_dict(
        annotations["PARATOPE_DICT_REFORMAT"]
    )

    return annotations


def paratopes_for_df_both_chains(
    df: pd.DataFrame,
    vl_df: pd.DataFrame,
    paratope_residue_threshold: Optional[float] = 0.67,
    both_chains: Optional[bool] = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    Calculates paratopes for VH or VH and VL,
    formats output into a single dataframe
    """

    df = parapred_for_df(df, paratope_residue_threshold=paratope_residue_threshold)
    df = paratopes_for_df(df, paratope_residue_threshold=paratope_residue_threshold)
    if both_chains:
        vl_df = parapred_for_df(
            vl_df, paratope_residue_threshold=paratope_residue_threshold
        )
        vl_df = paratopes_for_df(
            vl_df, paratope_residue_threshold=paratope_residue_threshold
        )

        vl_df = vl_df[
            [
                "CDR1",
                "CDR2",
                "CDR3",
                "PARATOPE_DICT",
                "PARATOPE_DICT_NUMBERED",
                "PARATOPE",
            ]
        ]

        # and 'L' and 'H' prefix for VL and VL col names
        vl_df.columns = ["L" + i for i in vl_df.columns]
        df.rename(
            columns={"CDR1": "HCDR1", "CDR2": "HCDR2", "CDR3": "HCDR3"}, inplace=True
        )
        # merge both dataframes
        df = df.merge(vl_df, how="left", left_index=True, right_index=True)
        # exclude sequences where parapred has failed for VH paratope or VL paratope
        nan_df3 = df[df["PARATOPE"].isnull() | df["LPARATOPE"].isnull()]
        df = df[df["PARATOPE"].notnull() | df["LPARATOPE"].notnull()]

        # rename paratope dict keys
        df["PARATOPE_DICT"] = [rename_dict_keys(i, "H") for i in df["PARATOPE_DICT"]]
        df["LPARATOPE_DICT"] = [rename_dict_keys(i, "L") for i in df["LPARATOPE_DICT"]]
        df["PARATOPE_DICT_NUMBERED"] = [
            rename_dict_keys(i, "H") for i in df["PARATOPE_DICT_NUMBERED"]
        ]
        df["LPARATOPE_DICT_NUMBERED"] = [
            rename_dict_keys(i, "L") for i in df["LPARATOPE_DICT_NUMBERED"]
        ]

        # merge paratope columns
        df["PARATOPE_DICT"] = [
            {**dict1, **dict2}
            for dict1, dict2 in zip(df["PARATOPE_DICT"], df["LPARATOPE_DICT"])
        ]
        df["PARATOPE_DICT_NUMBERED"] = [
            {**dict1, **dict2}
            for dict1, dict2 in zip(
                df["PARATOPE_DICT_NUMBERED"], df["LPARATOPE_DICT_NUMBERED"]
            )
        ]
        df["PARATOPE"] = df["PARATOPE"] + " " * 4 + df["LPARATOPE"]
    else:
        # exclude sequences where parapred has failed
        nan_df3 = df[df["PARATOPE"].isnull()]
        df = df[df["PARATOPE"].notnull()]

    return df, nan_df3


def tokenize_and_reformat(
    df: pd.DataFrame,
    structural_equivalence: Optional[bool] = True,
    tokenize: Optional[bool] = False,
) -> pd.DataFrame:

    """
    Reformats paratope column in the dataframe;
    Counts paratope length;
    Optionally tokenizes residues;
    """

    paratope_dict_col = "PARATOPE_DICT"
    if structural_equivalence is True:
        paratope_dict_col = "PARATOPE_DICT_NUMBERED"

    # reformatting paratope dict column
    if tokenize:
        residue_token_dict = get_residue_token_dict()
        df["PARATOPE_DICT_REFORMAT"] = [
            {
                cdr: {
                    str(residue[0]): residue_token_dict[residue[1]] for residue in value
                }
                for cdr, value in paratope_dict.items()
            }
            for paratope_dict in df[paratope_dict_col]
        ]
    else:
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

    return df
