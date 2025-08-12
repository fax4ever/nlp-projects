import ast
from pathlib import Path
import numpy as np
import pytest
from datasets import load_dataset, DatasetDict
from sentence_splitter_embedding_model.splitter_with_encoder import SplitterWithEncoder
import evaluate
from util.utils import compute_f1_metric

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


def test_try_eval():
    metric = evaluate.load("seqeval")
    a = [[str(num) for num in [1, 2, 3]]]
    b = [[str(num) for num in [1, 2, 3]]]
    predictions = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    references = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    print(metric.compute(predictions=predictions, references=references))


def test_load_data_txt_from_same_directory():
    data_path = Path(__file__).parent / "data.txt"
    text = data_path.read_text(encoding="utf-8")
    data = ast.literal_eval(text)
    assert isinstance(data, list)
    assert len(data) == 6

    pred = data[0][0]
    ref = data[1][0]
    assert pred is not None
    assert ref is not None
    assert len(pred) == len(ref)

    val = compute_f1_metric(data[0], data[1])
    val2 = compute_f1_metric(data[2], data[3])
    val3 = compute_f1_metric(data[4], data[5])
    print(val, val2, val3)

def test_simple_f1_metric():
    metric = evaluate.load("f1", average="binary")
    metric.add_batch(predictions=[1, 2, 2], references=[1, 2, 2])
    metric.add_batch(predictions=[2, 2, 2], references=[1, 2, 2])
    ciao = metric.compute()
    print(ciao)



    




    
