import os

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

    def push(self, repo):
        self.model.push_to_hub(repo, token=os.environ['HUGGINGFACE_TOKEN'])