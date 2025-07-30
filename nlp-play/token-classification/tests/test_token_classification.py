import pytest
from token_classification.token_classification import TokenClassification
from token_classification.token_classification_inference import TokenClassificationInference
from transformers import pipeline


@pytest.fixture
def test_subject():
    return TokenClassification()


@pytest.fixture
def test_subject_inference():
    return TokenClassificationInference()


def test_encoder_splitter(test_subject: TokenClassification):
    batch = test_subject.data_collator([test_subject.tokenized_dataset["train"][i] for i in range(2)])
    assert batch["labels"] is not None


def test_ext():
    model_checkpoint = "fax4ever/bert-finetuned-ner"
    token_classifier = pipeline(
        "token-classification", model=model_checkpoint, aggregation_strategy="simple"
    )
    result = token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
    print(result)


def test_inference(test_subject_inference: TokenClassificationInference):
    result = test_subject_inference.token_classify("My name is Sylvain and I work at Hugging Face in Brooklyn.")
    assert len(result) == 3
    assert result[0]["entity_group"] == "PER"
    assert result[0]["score"] > 0.8
    assert result[1]["entity_group"] == "ORG"
    assert result[1]["score"] > 0.8
    assert result[2]["entity_group"] == "LOC"
    assert result[2]["score"] > 0.8

