import torch


class EncoderHyperParams:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device:", self.device)
        self.embedding_model_name = "xlm-roberta-base"
        self.epochs = 8
        self.batch_size = 32
        self.warmup_steps = 500
        self.learning_rate = 5e-5
        self.weight_decay = 0.05