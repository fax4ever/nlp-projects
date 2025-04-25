import torch, csv
from torch.utils.data import DataLoader

from seed import set_seed
from processed_dataset import ProcessedDataset
from multi_modal_model import MultiModalModel

def output_label(label):
    if label == 0:
        return 'cultural agnostic'
    if label == 1:
        return 'cultural representative'
    if label == 2:
        return 'cultural exclusive'
    raise ValueError('label not suppoerted: ' + label)

def main():
    set_seed(42)
    model = MultiModalModel.from_pretrained("fax4ever/culturalitems-no-transformer")
    matching = 0

    with open('no-transformer-inference.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["entity", "true label", "prediction", "correct"]
        writer.writerow(field)
        with torch.no_grad():
            validation = ProcessedDataset().validation()
            for entity in DataLoader(validation):
                prediction = model(entity).detach().clone().argmax(dim=1).numpy()[0]
                true_label = entity['output_label'].numpy()[0]
                match = prediction == true_label
                if match:
                    matching = matching + 1
                base_ = entity['base'][0]
                writer.writerow([base_, output_label(true_label), output_label(prediction), match])

    print('matched', matching, 'on', len(validation), '(', matching/len(validation), ')')

if __name__ == "__main__":
    main()