import random
import numpy as np
import torch as T

from utils.logger import SingletonLogger


logger_instance = SingletonLogger()


def set_seed(seed: int = 42, deterministic: bool = True):
    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    T.manual_seed(seed)  # type: ignore
    T.cuda.manual_seed(seed)
    T.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Ensure deterministic behavior in convolution operations
    T.backends.cudnn.deterministic = deterministic
    T.backends.cudnn.benchmark = False

    logger_instance.logger.info(f"Global seed set to {seed}")

def seed_worker(worker_id: int):
    worker_seed = T.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
