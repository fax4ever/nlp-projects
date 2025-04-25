
from seed import set_seed
from processed_dataset import ProcessedDataset
from multi_modal_model import MultiModalModel

def main():
    set_seed(42)
    validation_set = ProcessedDataset().processed_validation_set
    model = MultiModalModel.from_pretrained("fax4ever/culturalitems-no-transformer")
    print(model)

if __name__ == "__main__":
    main()