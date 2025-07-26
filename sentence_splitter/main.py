import pandas as pd
from util.utils import set_seed
from sentence_splitter_embedding_model.encoder_splitter import EncoderSplitter


def main():
    set_seed()
    train = pd.read_csv("data/manzoni_train_tokens.csv")  # token,label
    print("Train DataFrame")
    print(train)
    validation = pd.read_csv("data/manzoni_dev_tokens.csv")  # token,label
    print("Validation DataFrame")
    print(validation)

    encoder_splitter = EncoderSplitter(train, validation)


if __name__ == "__main__":
    main()
