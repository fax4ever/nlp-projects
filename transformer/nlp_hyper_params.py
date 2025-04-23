import torch
import evaluate
import numpy as np
from transformers import set_seed

# Metrics
def compute_metrics(eval_pred):
    load_accuracy = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    return {"accuracy": accuracy}

class NLPHyperParams:
    def __init__(self):
        set_seed(42)
        self.language_model_name = "distilbert-base-uncased" #Distil-BERT
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.weight_decay = 0.001 # we could use e.g. 0.01 in case of very low and very high amount of data for regularization
        self.epochs = 30
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device: ", self.device)

