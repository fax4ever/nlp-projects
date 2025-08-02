from datasets import load_dataset
from sentence_splitter_embedding_model.splitter_with_encoder import SplitterWithEncoder


def main():
    dataset_dict = load_dataset("fax4ever/manzoni")
    splitter = SplitterWithEncoder()
    splitter.train(dataset_dict)
    

if __name__ == "__main__":
    main()
