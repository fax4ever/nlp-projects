import pytest
from token_classification.token_classification import TokenClassification


@pytest.fixture
def name():
    return "Mike"


@pytest.fixture
def test_subject(name):
    return TokenClassification(name)


def test_encoder_splitter(test_subject: TokenClassification, name):
    assert test_subject.hello() == "ciao " + name
    tokenized_inputs = test_subject.tokenize_and_align_labels(
        test_subject.dataset["train"]
    )
    assert tokenized_inputs is not None
