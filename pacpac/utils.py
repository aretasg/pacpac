# Author: Aretas Gaspariunas

from typing import Dict, Any

from numba.typed import Dict as numbaDict
from numba import types


def convert_to_typed_numba_dict(
    input_dict: Dict[str, Dict[str, str]]
) -> Dict[str, Dict[str, str]]:

    # https://github.com/numba/numba/issues/6191#issuecomment-684022879
    """
    Converts nested Python dictionary to a typed numba dictionary
    """

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


def rename_dict_keys(input_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:

    """
    Creates a copy of an input_dict with keys carrying the prefix specified
    """

    current_keys = input_dict.keys()
    new_keys = [prefix + i for i in current_keys]
    new_dict = {
        new: input_dict[current] for current, new in zip(current_keys, new_keys)
    }

    return new_dict
