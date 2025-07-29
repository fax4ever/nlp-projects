import pytest
from token_classification.token_classification import TokenClassification


@pytest.fixture
def test_subject():
    return TokenClassification()


def test_encoder_splitter(test_subject: TokenClassification):
    batch = test_subject.data_collator([test_subject.tokenized_dataset["train"][i] for i in range(2)])
    assert batch["labels"] is not None
