from transformers import AutoTokenizer
from datasets import DatasetDict


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


class SplitterWithEncoder:

    def __init__(self):
        pass

    def train(self, dataset_dict:DatasetDict):
        base_embedding_model_name = "bert-base-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(base_embedding_model_name)
        
        self.tokenized_dataset_dict = dataset_dict.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset_dict["train"].column_names,
            batch_size=256,
        )
        label_names = [0, 1]

    def tokenize_and_align_labels(self, items):
        tokenized_inputs = self.tokenizer(
            items["token"], is_split_into_words=True
        )
        tokenized_inputs["labels"] = align_labels_with_tokens(items["label"], tokenized_inputs.word_ids(0))
        return tokenized_inputs

    def load_model(self):
        pass
