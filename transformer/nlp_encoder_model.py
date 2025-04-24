import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from nlp_hyper_params import NLPHyperParams

class NLPEncoderModel:
    def __init__(self, params: NLPHyperParams):
        self.params = params
        ## Initialize the model
        self.model = AutoModelForSequenceClassification.from_pretrained(params.language_model_name, ignore_mismatched_sizes=True,
            output_attentions=False, output_hidden_states=False, num_labels=3) # number of the classes
        # Load the pretrained tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(params.language_model_name)
        # Set the data collator
        # Function used to prepare the data before the training.
        # the data collator function used here apply a zero-padding on the elements in the batch
        # the padding is needed to have a "full" form of the batches
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def predict(self, text, max_length=128):
        self.model.eval()
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.params.max_length, padding='max_length', truncation=True)
        input_ids = encoding['input_ids'].to(self.params.device)
        attention_mask = encoding['attention_mask'].to(self.params.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
        return preds.item()