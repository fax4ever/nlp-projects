import pytest


@pytest.fixture
def hello():
    return 'ciao'


def test_encoder_splitter(hello):
    assert hello is not None
