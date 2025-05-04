import csv, sys
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from inference_model import InferenceModel
sys.path.insert(1, sys.path[0].replace("transformer", "no_transformer"))
from nlp_dataset import NLPDataset

def number_to_label(label):
    if label == 0:
        return 'cultural agnostic'
    if label == 1:
        return 'cultural representative'
    if label == 2:
        return 'cultural exclusive'
    raise ValueError('label not suppoerted: ' + label)

def main():
    nlp_dataset = NLPDataset()
    model = InferenceModel("fax4ever/culturalitems-roberta-base-5", "roberta-base")

    matching = 0
    predictions = []
    labels = []
    for index, item in enumerate(nlp_dataset.validation_set):
        prediction = model.predict(item.description, item.wiki_text)
        predicted_label = number_to_label(prediction)
        predictions.append(predicted_label)
        labels.append(item.label)
        match = predicted_label == item.label
        if match:
            matching = matching + 1
        if (index + 1) % 10 == 0:
            print('inference of the validation set: ', index + 1, "/", len(nlp_dataset.validation_set))
            print('matched', matching, 'on', index + 1, '(', matching / (index + 1), ')')
    print('inference of the validation: completed')
    print('matched', matching, 'on', len(nlp_dataset.validation_set), '(', matching / len(nlp_dataset.validation_set), ')')

    cm = confusion_matrix(labels, predictions)
    ConfusionMatrixDisplay(cm).plot()
    print('f1', f1_score(labels, predictions, average='macro'))
    print('recall', recall_score(labels, predictions, average='macro'))
    print('precision', precision_score(labels, predictions, average='macro'))
    print('accuracy_score', accuracy_score(labels, predictions))

    with open('Lost_in_Language_Recognition_output_roberta.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ['item', 'name', 'label']
        writer.writerow(field)
        for index, item in enumerate(nlp_dataset.test_set):
            prediction = model.predict(item.description, item.wiki_text)
            writer.writerow(["http://www.wikidata.org/entity/" + item.entity_id, item.name, number_to_label(prediction)])
            if (index + 1) % 10 == 0:
                print('inference of the test set: ', index + 1, "/", len(nlp_dataset.test_set))
        print('inference of the test set: completed')

if __name__ == "__main__":
    main()