import itertools
import os
import shutil


def check_data_exists(root_dir: str = "data", subdir: str = "raw"):
    path = os.path.join(root_dir, subdir)
    return os.path.exists(path) and len(os.listdir(path)) > 0


def remove_dir_with_content(path: str):
    """Remove directory with its content if it exists."""
    if os.path.exists(path):
        shutil.rmtree(path)

def make_dirs(*args: list[str]):
    list_ = itertools.product(*args)
    for path_list in list_:
        path = os.path.join(*path_list)
        os.makedirs(path, exist_ok=True)
        
    return list_

def extract_file_paths(directory: str, file_name: str) -> list[str]:
    list_: list[str] = []
    for path, _, file in os.walk(directory):
        if file_name in file:
            list_.append(os.path.join(path, file_name))
    return list_
