from datasets import load_dataset
from sentence_splitter_embedding_model.splitter_with_encoder import SplitterWithEncoder


def main():
    dataset_dict = load_dataset("fax4ever/manzoni-192")
    splitter = SplitterWithEncoder()
    splitter.tokenize_dataset(dataset_dict, "bert-base-cased")
    splitter.train()
    

if __name__ == "__main__":
    main()
