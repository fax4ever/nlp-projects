from datasets import load_dataset
from transformers import AutoTokenizer


class TokenClassification:
    def __init__(self, name):
        self.name = name
        self.dataset = load_dataset("conll2003", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        print("tokenizer.is_fast", self.tokenizer.is_fast)

    def hello(self):
        return "ciao " + self.name

    def tokenize_and_align_labels(self, items):
        tokenized_inputs = self.tokenizer(
            items["tokens"], truncation=True, is_split_into_words=True
        )
        all_labels = items["ner_tags"]
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
