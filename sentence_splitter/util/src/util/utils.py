import torch
import random
import numpy as np
from typing import Iterable, List, Sequence
import evaluate

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


def map_nested_ints_to_strings(values: Iterable[Iterable[int]]) -> List[List[str]]:
    return [
        [str(item) for item in inner_iter]
        for inner_iter in values
    ]

def remove_minus100(reference_batch: Iterable) -> List:
    return [item for item in reference_batch if item != -100 and item != '-100']


def compute_f1_metric(predictions: List[List], references: List[List]) -> float:
    metric = evaluate.load("f1", average="binary")

    for i, prediction_batch in enumerate(predictions):
        reference_batch = references[i]


    metric.add_batch(predictions=predictions, references=references)
    return metric.compute()
