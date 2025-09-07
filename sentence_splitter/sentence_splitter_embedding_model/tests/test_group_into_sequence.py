import pandas as pd
import pytest

from sentence_splitter_embedding_model.group_into_sequence import group_into_sequences


@pytest.fixture
def df_simple():
    return pd.DataFrame({'token': ['Hello', 'world', 'this', 'is', 'a', 'test', 'blah'], 'label': [1, 0, 1, 0, 1, 0, 0]})


@pytest.fixture
def pandas_df():
    return pd.read_csv("data/OOD_test.csv", sep=';')


def test_group_into_sequences(df_simple):
    result = group_into_sequences(df_simple)
    expected = {
        'tokens': [['Hello', 'world', 'this'], ['is', 'a', 'test'], ['blah']],
        'labels': [[1, 0, 1], [0, 1, 0], [0]]
    }
    assert result == expected


def test_group_into_sequences_pandas(pandas_df):
    result = group_into_sequences(pandas_df)
    all_words = [word for sequence in result['tokens'] for word in sequence]
    all_labels = [label for sequence in result['labels'] for label in sequence]
    assert len(all_words) == len(all_labels)
    assert all(isinstance(word, str) for word in all_words)
    assert all(isinstance(label, int) for label in all_labels)