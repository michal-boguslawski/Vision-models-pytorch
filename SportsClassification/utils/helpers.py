from collections.abc import Callable
from decimal import Decimal
import inspect
import json
import os
import secrets
import string
from typing import Dict, Any, Optional, Mapping


def filter_kwargs(cls: Callable[..., Any], config: Optional[dict[str, Any]]) -> Dict[str, Any]:
    config = config or {}
    valid_params = inspect.signature(cls).parameters
    filtered_config = {k: v for k, v in config.items() if k in valid_params}
    return filtered_config


def append_dict_to_dict(dict_: Mapping[str, float | list[float]], new_dict_: Mapping[str, float | list[float | int]]):
    result: dict[str, list[float]] = {}
    # Convert existing dict_ to list values
    for k, v in dict_.items():
        result[k] = v if isinstance(v, list) else [v]

    # Append new metrics
    for key, values in new_dict_.items():
        values = values if isinstance(values, list) else [values]
        if key in result:
            result[key].extend(values)
        else:
            result[key] = values
    return result


def load_dict_and_append(path: str, new_dict_: dict[str, list[float] | float]):
    # Load existing dict of lists
    dict_: dict[str, list[float]] = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            dict_ = json.load(f)


    dict_ = append_dict_to_dict(dict_, new_dict_=new_dict_)

    return dict_

def generate_random_name(length: int = 12) -> str:
    # Define allowed characters
    alphabet = string.ascii_lowercase + string.digits
    
    # Ensure password has at least one of each type (optional, but good practice)
    while True:
        random_name = ''.join(secrets.choice(alphabet) for _ in range(length))
        if (any(c.islower() for c in random_name)
            and any(c.isdigit() for c in random_name)):
            return random_name

def convert_floats_to_decimal(d):
    """Recursively convert all floats in dict/list to Decimal."""
    if isinstance(d, dict):
        return {k: convert_floats_to_decimal(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_floats_to_decimal(v) for v in d]
    elif isinstance(d, float):
        return Decimal(str(d))  # use str to preserve precision
    else:
        return d
