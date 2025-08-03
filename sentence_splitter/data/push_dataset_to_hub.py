from datasets import Dataset, DatasetDict
import os
import pandas as pd


SIZE = 256


def group_into_sequences(df, seq_len=SIZE):
    tokens = df['token'].tolist()
    labels = df['label'].tolist()
    
    # Group into sequences of seq_len
    token_seqs = [tokens[i:i+seq_len] for i in range(0, len(tokens), seq_len) if len(tokens[i:i+seq_len]) == seq_len]
    label_seqs = [labels[i:i+seq_len] for i in range(0, len(labels), seq_len) if len(labels[i:i+seq_len]) == seq_len]
    
    return {'tokens': token_seqs, 'labels': label_seqs}

def main():
    train = pd.read_csv("data/manzoni_train_tokens.csv")  # token,label
    validation = pd.read_csv("data/manzoni_dev_tokens.csv")  # token,label
    
    # Group into sequences of SIZE
    train_grouped = group_into_sequences(train)
    validation_grouped = group_into_sequences(validation)
    
    print(f"Train: {len(train_grouped['tokens'])} sequences of {SIZE} tokens each")
    print(f"Validation: {len(validation_grouped['tokens'])} sequences of {SIZE} tokens each")

    train_dataset = Dataset.from_dict(train_grouped)
    validation_dataset = Dataset.from_dict(validation_grouped)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset  # Using 'validation' as the standard name
    })
    dataset_dict.push_to_hub(f"fax4ever/manzoni-{SIZE}", token=os.getenv("HF_TOKEN"))


if __name__ == "__main__":
    main()
