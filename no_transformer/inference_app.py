import torch, csv
from torch.utils.data import DataLoader

from seed import set_seed
from processed_dataset import ProcessedDataset
from multi_modal_model import MultiModalModel

def number_to_label(label):
    if label == 0:
        return 'cultural agnostic'
    if label == 1:
        return 'cultural representative'
    if label == 2:
        return 'cultural exclusive'
    raise ValueError('label not suppoerted: ' + label)

def main():
    set_seed(42)
    model = MultiModalModel.from_pretrained("fax4ever/no-transformer-test")
    processed_dataset = ProcessedDataset()

    matching = 0
    with torch.no_grad():
        validation = processed_dataset.validation()
        for entity in DataLoader(validation):
            prediction = model(entity).detach().clone().argmax(dim=1).numpy()[0]
            true_label = entity['output_label'].numpy()[0]
            match = prediction == true_label
            if match:
                matching = matching + 1
    print('matched', matching, 'on', len(validation), '(', matching/len(validation), ')')

    with open('Lost_in_Language_Recognition_output_multimodalnn.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ['item', 'name', 'label']
        writer.writerow(field)
        with torch.no_grad():
            test = processed_dataset.test()
            for entity in DataLoader(test):
                prediction = model(entity).detach().clone().argmax(dim=1).numpy()[0]
                writer.writerow(["http://www.wikidata.org/entity/" + entity['entity_id'][0], entity['name'][0], number_to_label(prediction)])

if __name__ == "__main__":
    main()