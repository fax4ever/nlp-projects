from tqdm.auto import tqdm
from multi_modal_model import MultiModalModel
from nlp_hyper_params import NLPHyperParams

class NLPTrainer:
    def __init__(self, model: MultiModalModel, optimizer, loss_function, device, params: NLPHyperParams):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.params = params

    def train(self, train_dataset):
        train_loss = 0.0
        for epoch in range(self.params.epochs):
            epoch_loss = 0.0
            num_batches = 0
            for step, dataset_items in tqdm(enumerate(train_dataset), desc="Batch", leave=False):
                targets = dataset_items['output_label'].to(self.device)
                output_distribution = self.model(dataset_items, self.device)
                loss = self.loss_function(output_distribution, targets)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_loss += loss.item()
                num_batches += 1
            avg_epoch_loss = epoch_loss / num_batches
            print(f'Epoch: {epoch} avg loss = {avg_epoch_loss:.4f}')
            train_loss += avg_epoch_loss