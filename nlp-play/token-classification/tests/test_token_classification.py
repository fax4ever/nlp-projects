import pytest
from token_classification.token_classification import TokenClassification


@pytest.fixture
def test_subject():
    return TokenClassification()


def test_encoder_splitter(test_subject: TokenClassification):
    assert test_subject.tokenized_dataset["train"][0:1] is not None
