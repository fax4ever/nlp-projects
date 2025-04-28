import csv

from inference_model import InferenceModel
from wiki_dataset import WikiDataset

def number_to_label(label):
    if label == 0:
        return 'cultural agnostic'
    if label == 1:
        return 'cultural representative'
    if label == 2:
        return 'cultural exclusive'
    raise ValueError('label not suppoerted: ' + label)

def main():
    dataset = WikiDataset()
    model = InferenceModel("fax4ever/culturalitems-roberta-base-5", "roberta-base")

    tokenized_datasets = dataset.tokenize(model.tokenizer)
    print(tokenized_datasets)
    validation_ = tokenized_datasets["validation"]

    matching = 0
    matching_ds = 0
    size = len(validation_)
    with open('transformer-inference.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["item", "true label", "prediction", "prediction-ds", "correct", "correct-ds"]
        writer.writerow(field)
        for index, item in enumerate(validation_):
            p, p_ds = model.predict_text(item["description"], item["wiki_text"], item["input_ids"], item["attention_mask"])
            true_label = item["label"]
            match = p == true_label
            if match:
                matching = matching + 1
            match_ds = p_ds == true_label
            if match_ds:
                matching_ds = matching_ds + 1
            writer.writerow([item["item"], number_to_label(true_label), number_to_label(p), number_to_label(p_ds), match, match_ds])
            if (index + 1) % 10 == 0:
                print('inference: ', index + 1, "/", size)
                print('matched', matching, 'on', index + 1, '(', matching / (index + 1), ')')
                print('matched', matching_ds, 'on', index + 1, '(', matching_ds / (index + 1), ')')
    print('inference: completed')
    print('matched', matching, 'on', size, '(', matching / size, ')')
    print('matched', matching_ds, 'on', size, '(', matching_ds / size, ')')

if __name__ == "__main__":
    main()