from processed_dataset import ProcessedDataset
from seed import set_seed

def main():
    set_seed(42)
    dataset = ProcessedDataset()
    print("dataset loaded: " + str(dataset))
    pass

if __name__ == "__main__":
    main()