from sentence_splitter_llm.dataset_factory import conversations_dataset

def test_conversations_dataset():
    dataset = conversations_dataset()

    zero_train_example = dataset['train'][60]
    assert zero_train_example is not None
