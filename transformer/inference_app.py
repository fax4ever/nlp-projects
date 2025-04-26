from transformer.inference_model import InferenceModel
from transformer.wiki_dataset import WikiDataset

def main():
    dataset = WikiDataset()
    model = InferenceModel("../culturalitems-distilbert", "distilbert-base-uncased")

    tokenized_datasets = dataset.tokenize(model.tokenizer)
    print(tokenized_datasets)
    validation_ = tokenized_datasets["validation"]
    for item in validation_:
        prediction = model.predict_text(item["description"], item["wiki_text"], item["input_ids"], item["attention_mask"])
        print(prediction)

if __name__ == "__main__":
    main()