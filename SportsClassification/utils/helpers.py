import inspect
import json
import os
from collections.abc import Callable
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
