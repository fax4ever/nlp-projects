import pytest
from datasets import load_dataset, DatasetDict
from sentence_splitter_embedding_model.splitter_with_encoder import SplitterWithEncoder


@pytest.fixture
def dataset_dict():
    return load_dataset("fax4ever/manzoni-192")


@pytest.fixture
def splitter():
    return SplitterWithEncoder()


def test_splitter_base_bert(splitter:SplitterWithEncoder, dataset_dict:DatasetDict):
    splitter.tokenize_dataset(dataset_dict, "bert-base-cased")
    assert splitter.tokenized_dataset_dict is not None

    labels_0 = splitter.tokenized_dataset_dict['train']['labels'][0]
    input_ids_0 = splitter.tokenized_dataset_dict['train']['input_ids'][0]
    assert len(labels_0) == len(input_ids_0)
    assert len(labels_0) <= 512


def test_splitter_modern_bert(splitter:SplitterWithEncoder, dataset_dict:DatasetDict):
    splitter.tokenize_dataset(dataset_dict, "DeepMount00/ModernBERT-base-ita")
    assert splitter.tokenized_dataset_dict is not None

    labels_0 = splitter.tokenized_dataset_dict['train']['labels'][0]
    input_ids_0 = splitter.tokenized_dataset_dict['train']['input_ids'][0]
    assert len(labels_0) == len(input_ids_0)
    assert len(labels_0) <= 512


def test_splitter_italian_bert(splitter:SplitterWithEncoder, dataset_dict:DatasetDict):
    splitter.tokenize_dataset(dataset_dict, "dbmdz/bert-base-italian-xxl-cased")
    assert splitter.tokenized_dataset_dict is not None

    labels_0 = splitter.tokenized_dataset_dict['train']['labels'][0]
    input_ids_0 = splitter.tokenized_dataset_dict['train']['input_ids'][0]
    assert len(labels_0) == len(input_ids_0)
    assert len(labels_0) <= 512
    




    
