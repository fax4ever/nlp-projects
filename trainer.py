from tqdm.auto import tqdm
from multi_modal_model import MultiModalModel
from nlp_hyper_params import NLPHyperParams

class NLPTrainer:
    def __init__(self, model: MultiModalModel, optimizer, criterion, device, params: NLPHyperParams):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.params = params

    def train(self, train_dataset):
        train_loss = 0.0
        for epoch in range(self.params.epochs):
            epoch_loss = 0.0
            num_batches = 0
            for step, sample in tqdm(enumerate(train_dataset), desc="Batch", leave=False):
                pass