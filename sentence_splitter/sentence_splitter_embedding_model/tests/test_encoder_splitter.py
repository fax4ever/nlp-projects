import pytest
import pandas as pd
from sentence_splitter_embedding_model.encoder_splitter import EncoderSplitter


@pytest.fixture
def train():
    return pd.DataFrame({'token': ['I', 'am', 'Fabio', '.', 'Going', 'to', 'see', 'Mike', '.'],
                         'label': [0, 0, 0, 1, 0, 0, 0, 0, 1]})


@pytest.fixture
def dev():
    return pd.DataFrame({'token': ['I', 'am', 'Fabio', '.'], 'label': [0, 0, 0, 1]})


@pytest.fixture
def encoder_splitter(train, dev):
    return EncoderSplitter(train, dev)


def test_encoder_splitter(encoder_splitter):
    assert encoder_splitter is not None
