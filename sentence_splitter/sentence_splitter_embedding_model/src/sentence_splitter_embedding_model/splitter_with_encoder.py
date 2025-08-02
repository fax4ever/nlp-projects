from transformers import AutoTokenizer


class SplitterWithEncoder:

    def __init__(self):
        pass

    def train(self, dataset_dict):
        base_embedding_model_name = "bert-base-cased"
        tokenizer = AutoTokenizer.from_pretrained(base_embedding_model_name)

    def load_model(self):
        pass
