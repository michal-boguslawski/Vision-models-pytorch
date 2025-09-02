import inspect
import json
import os


def check_data_exists(root_dir: str = "data", subdir: str = "raw", *args, **kwargs):
    path = os.path.join(root_dir, subdir)
    return os.path.exists(path) and len(os.listdir(path)) > 0


def filter_kwargs(cls, config: dict):
    valid_params = inspect.signature(cls).parameters
    filtered_config = {k: v for k, v in config.items() if k in valid_params}
    return filtered_config


def append_dict_to_dict(dict_: dict, new_dict_: dict):
    # Append new metrics
    for key, values in new_dict_.items():
        values = values if isinstance(values, list) else [values]
        if key in dict_:
            dict_[key].extend(values)
        else:
            dict_[key] = values
    return dict_


def load_dict_and_append(path: str, new_dict_: dict):
    # Load existing dict of lists
    if os.path.exists(path):
        with open(path, "r") as f:
            dict_ = json.load(f)
    else:
        dict_ = {}


    dict_ = append_dict_to_dict(dict_, new_dict_=new_dict_)

    return dict_
