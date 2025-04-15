from processed_dataset import ProcessedDataset
from text_classifier import TextClassifier

def main():
    dataset = ProcessedDataset()
    print("dataset loaded: " + str(dataset))
    model = TextClassifier(input_size=dataset.desc_dictionary.length())
    pass

if __name__ == "__main__":
    main()