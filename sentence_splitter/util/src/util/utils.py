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


def compute_f1_metric(predictions: Iterable[Iterable], references: Iterable[Iterable]) -> float:
    metric = evaluate.load("f1", average="binary")

    assert len(predictions) == len(references)
    for i, prediction_batch in enumerate(predictions):
        reference_batch = references[i]
        
        prediction_valid_batch = []
        reference_valid_batch = []
        trailing = False

        assert len(prediction_batch) == len(reference_batch)
        for prediction, reference in zip(prediction_batch, reference_batch):
            if reference == -100 or reference == '-100':
                trailing = True
                continue
            else:
                assert not trailing
                reference_valid_batch.append(reference)
                prediction_valid_batch.append(prediction)

        metric.add_batch(predictions=prediction_valid_batch, references=reference_valid_batch)
    return metric.compute()
