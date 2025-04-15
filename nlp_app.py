from dataset_nlp import DatasetNLP
from text_classifier import TextClassifier

def main():
    dataset = DatasetNLP()
    print("dataset loaded: " + str(dataset))
    model = TextClassifier(input_size=dataset.desc_dictionary.length())
    pass

if __name__ == "__main__":
    main()