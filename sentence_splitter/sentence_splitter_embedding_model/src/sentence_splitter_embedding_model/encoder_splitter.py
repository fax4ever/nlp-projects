from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from sentence_splitter_embedding_model.encoder_hyperparams import EncoderHyperParams
from sentence_splitter_embedding_model.encoder_training import EncoderTrainer

class EncoderSplitter:

    def __init__(self, train_dataset, eval_dataset):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.params = EncoderHyperParams()
        self.model = AutoModelForSequenceClassification.from_pretrained(self.params.embedding_model_name,
                                                                        ignore_mismatched_sizes=True,
                                                                        output_attentions=False,
                                                                        output_hidden_states=False, num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained(self.params.embedding_model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.trainer = EncoderTrainer(self.params, self.model, self.tokenizer, self.data_collator, train_dataset, eval_dataset)