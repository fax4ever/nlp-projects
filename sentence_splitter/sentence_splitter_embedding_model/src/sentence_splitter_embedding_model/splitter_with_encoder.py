from transformers import AutoTokenizer
from datasets import DatasetDict

END_OF_SENTENCE = 1
NOT_END_OF_SENTENCE = 0
LABEL_FOR_START_END_OF_SEQUENCE = NOT_END_OF_SENTENCE

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
            batch_size=128,
        )
        label_names = [NOT_END_OF_SENTENCE, END_OF_SENTENCE]

    def tokenize_and_align_labels(self, items):
        tokenized_inputs = self.tokenizer(
            items["tokens"], is_split_into_words=True
        )

        all_labels = items["labels"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    def align_labels_with_tokens(self, labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = LABEL_FOR_START_END_OF_SEQUENCE if word_id is None else labels[word_id]
                new_labels.append(label)
            else:
                # Treat the same word never as end of sentence
                new_labels.append(NOT_END_OF_SENTENCE)
        return new_labels
