import itertools
import logging
import os
import shutil
from typing import Optional


_message_cache: list[str] = []

def cache_message(msg: str):
    _message_cache.append(msg)


def flush_cache():
    """Flush all cached messages to logger or print if no logger."""
    logger = logging.getLogger()
    for msg in _message_cache:
        if logger.hasHandlers():
            logger.info(msg)
        else:
            print(msg)
    _message_cache.clear()


def check_data_exists(root_dir: str = "data", subdir: str = "raw", *args, **kwargs):
    path = os.path.join(root_dir, subdir)
    return os.path.exists(path) and len(os.listdir(path)) > 0


def remove_dir_with_content(path: str):
    """Remove directory with its content if it exists."""
    logger = logging.getLogger()
    if os.path.exists(path):
        shutil.rmtree(path)
        msg = f"Folder {path} was removed with its entire content"
        if logger.hasHandlers():
            logger.info(msg)
        else:
            cache_message(msg)


def make_dirs(*args, **kwargs):
    logger = logging.getLogger()
    list_ = itertools.product(*args)
    for path_list in list_:
        path = os.path.join(*path_list)
        os.makedirs(path, exist_ok=True)
        msg = f"Path {path} created or existed"
        if logger.hasHandlers():
            logger.info(msg)
        else:
            cache_message(msg)
    return list_