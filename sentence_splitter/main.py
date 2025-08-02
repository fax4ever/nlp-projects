import pandas as pd
from util.utils import set_seed
from datasets import Dataset


def main():
    set_seed()
    train = pd.read_csv("data/manzoni_train_tokens.csv")  # token,label
    print("Train DataFrame")
    print(train)

    validation = pd.read_csv("data/manzoni_dev_tokens.csv")  # token,label
    print("Validation DataFrame")
    print(validation)

    train_dataset = Dataset.from_pandas(train)
    validation_dataset = Dataset.from_pandas(validation)
    


if __name__ == "__main__":
    main()
