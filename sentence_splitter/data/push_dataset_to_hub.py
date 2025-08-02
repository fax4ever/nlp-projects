from datasets import Dataset, DatasetDict
import os
import pandas as pd


def main():
    train = pd.read_csv("data/manzoni_train_tokens.csv")  # token,label
    print("Train DataFrame")
    print(train)

    validation = pd.read_csv("data/manzoni_dev_tokens.csv")  # token,label
    print("Validation DataFrame")
    print(validation)

    train_dataset = Dataset.from_pandas(train)
    validation_dataset = Dataset.from_pandas(validation)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset  # Using 'validation' as the standard name
    })
    dataset_dict.push_to_hub("fax4ever/manzoni", token=os.getenv("HF_TOKEN"))


if __name__ == "__main__":
    main()
