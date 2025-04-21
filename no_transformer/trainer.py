import statistics

import torch
from tqdm.auto import tqdm
from multi_modal_model import MultiModalModel
from nlp_hyper_params import NLPHyperParams

def compute_accuracy(predictions, labels):
    discrete_predictions = predictions.detach().clone()
    discrete_predictions = discrete_predictions.argmax(dim=1)
    return torch.mean(torch.Tensor([1 if p==l else 0 for p, l in list(zip(discrete_predictions.view(-1), labels.view(-1)))])).item()

class NLPTrainer:
    def __init__(self, model: MultiModalModel, optimizer, loss_function, device, params: NLPHyperParams):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.params = params

    def train(self, train_dataset, validation_dataset):
        print('Training...')
        train_loss = []
        train_accuracy = []
        valid_loss = []
        valid_accuracy = []

        for epoch in range(self.params.epochs):
            print(' Epoch {:03d}'.format(epoch + 1))
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            for step, dataset_items in tqdm(enumerate(train_dataset), desc="Batch", leave=False):
                targets = dataset_items['output_label'].to(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(dataset_items, self.device)
                sample_loss = self.loss_function(predictions, targets)
                sample_loss.backward()
                self.optimizer.step()
                epoch_loss += sample_loss.tolist()
                epoch_accuracy += compute_accuracy(predictions, targets)
            avg_epoch_loss = epoch_loss / len(train_dataset)
            avg_epoch_accuracy = epoch_accuracy / len(train_dataset)
            print('  [E: {:2d}] train loss = {:0.4f} | train accuracy = {:0.4f}'.format(epoch, avg_epoch_loss, avg_epoch_accuracy))
            valid_loss_epoch, valid_accuracy_epoch = self.evaluate(validation_dataset)
            print('  [E: {:2d}] valid loss = {:0.4f} | eval accuracy =. {:0.4f}'.format(epoch, valid_loss_epoch, valid_accuracy_epoch))
            train_loss.append(avg_epoch_loss)
            train_accuracy.append(avg_epoch_accuracy)
            valid_loss.append(valid_loss_epoch)
            valid_accuracy.append(valid_accuracy_epoch)
        print('... Done!')
        avg_epoch_loss = statistics.mean(train_loss)
        avg_epoch_accuracy = statistics.mean(train_accuracy)
        return {
            "avg_epoch_loss": avg_epoch_loss,
            "avg_epoch_accuracy": avg_epoch_accuracy,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "valid_loss": valid_loss,
            "valid_accuracy": valid_accuracy
        }

    def evaluate(self, valid_dataset):
        valid_loss = 0.0
        validation_accuracy = 0.0
        # no gradient updates here, we are at inference time!
        with torch.no_grad():
            for step, dataset_items in tqdm(enumerate(valid_dataset), desc="Batch", leave=False):
                targets = dataset_items['output_label'].to(self.device)
                predictions = self.model(dataset_items, self.device)
                sample_loss = self.loss_function(predictions, targets)
                valid_loss += sample_loss.tolist()
                validation_accuracy += compute_accuracy(predictions, targets)
        return valid_loss / len(valid_dataset), validation_accuracy / len(valid_dataset)

    def predict(self, x):
        return self.model(x).tolist()