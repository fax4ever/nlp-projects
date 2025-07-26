import torch
import random
import numpy as np


def set_seed(seed=777, total_determinism=False):
    seed = seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if total_determinism:
        torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
