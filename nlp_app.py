from nlp_hyper_params import NLPHyperParams
from processed_dataset import ProcessedDataset
from seed import set_seed

def main():
    set_seed(42)
    dataset = ProcessedDataset()
    print("dataset loaded: " + str(dataset))
    hyper_params = NLPHyperParams()
    hyper_params.compute(dataset.processed_training_set[0])
    pass

if __name__ == "__main__":
    main()