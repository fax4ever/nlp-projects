import torch


class EncoderHyperParams:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device:", self.device)