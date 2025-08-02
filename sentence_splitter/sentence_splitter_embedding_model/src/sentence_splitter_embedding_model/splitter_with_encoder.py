from transformers import AutoTokenizer


class SplitterWithEncoder:

    def __init__(self):
        pass

    def train(self, train_dataset, eval_dataset):
        base_embedding_model_name = "bert-base-cased"
        label_names = ["O", "1"]
        tokenizer = AutoTokenizer.from_pretrained(base_embedding_model_name)

    def load_model(self):
        pass
