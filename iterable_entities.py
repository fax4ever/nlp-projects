from torch.utils.data import IterableDataset
from processed_entity import ProcessedEntity

class IterableEntities(IterableDataset):
    def __init__(self, processed_entities: list[ProcessedEntity]):
        self.processed_entities = processed_entities

    def __iter__(self):
        for entity in self.processed_entities:
            yield entity.dataset_item()

