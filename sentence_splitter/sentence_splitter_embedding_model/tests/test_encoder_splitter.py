import pytest
import pandas as pd
from datasets import Dataset


@pytest.fixture
def train():
    return pd.DataFrame({'token': ['I', 'am', 'Fabio', '.', 'Going', 'to', 'see', 'Mike', '.'],
                         'label': [0, 0, 0, 1, 0, 0, 0, 0, 1]})


@pytest.fixture
def dev():
    return pd.DataFrame({'token': ['I', 'am', 'Fabio', '.'], 'label': [0, 0, 0, 1]})


@pytest.fixture
def train_dataset(train):
    return Dataset.from_pandas(train)


@pytest.fixture
def dev_dataset(dev):
    return Dataset.from_pandas(dev)


def test_train_and_evaluate(train_dataset, dev_dataset):
    assert train_dataset is not None
    assert dev_dataset is not None

