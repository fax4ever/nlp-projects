import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from multi_modal_model import MultiModalModel
from nlp_hyper_params import NLPHyperParams
from processed_dataset import ProcessedDataset
from seed import set_seed
from trainer import NLPTrainer


def main():
    set_seed(42)
    dataset = ProcessedDataset()
    print("dataset loaded: " + str(dataset))
    hyper_params = NLPHyperParams(dataset.processed_training_set[0])
    training_dataloader = DataLoader(dataset.training(), batch_size=hyper_params.batch_size)
    validation_dataloader = DataLoader(dataset.validation(), batch_size=hyper_params.batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultiModalModel(hyper_params)
    optimizer = optim.Adam(model.parameters(), lr=hyper_params.learning_rate)
    criterion = nn.CrossEntropyLoss()
    trainer = NLPTrainer(model, optimizer, criterion, device, hyper_params)
    trainer.train(training_dataloader)
    pass

if __name__ == "__main__":
    main()