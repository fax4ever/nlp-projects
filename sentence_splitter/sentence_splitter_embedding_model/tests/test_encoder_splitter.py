import pytest
from datasets import load_dataset, DatasetDict
from sentence_splitter_embedding_model.splitter_with_encoder import SplitterWithEncoder


@pytest.fixture
def dataset_dict():
    return load_dataset("fax4ever/manzoni")


@pytest.fixture
def splitter():
    return SplitterWithEncoder()


def test_splitter(splitter:SplitterWithEncoder, dataset_dict:DatasetDict):
    splitter.train(dataset_dict)
    assert splitter.tokenized_dataset_dict is not None




    
