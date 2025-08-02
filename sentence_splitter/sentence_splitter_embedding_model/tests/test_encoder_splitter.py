import pytest
from datasets import load_dataset
from sentence_splitter_embedding_model.splitter_with_encoder import SplitterWithEncoder


@pytest.fixture
def dataset_dict():
    return load_dataset("fax4ever/manzoni")


@pytest.fixture
def splitter():
    return SplitterWithEncoder()


def test_splitter(splitter, dataset_dict):
    splitter.train(dataset_dict)




    
