import pandas as pd
from processed_entity import ProcessedEntity
import numpy as np

class CategoryTable:
    def __init__(self):
        self.subcategories_entered = {}  # to avoid duplicates
        self.subcategories = []
        self.categories = []
        self.subcategory_to_id = None  # computed on build

    def include(self, processed_entity: ProcessedEntity):
        if processed_entity.subcategory in self.subcategories_entered:
            return
        self.subcategories_entered[processed_entity.subcategory] = True
        self.subcategories.append(processed_entity.subcategory)
        self.categories.append(processed_entity.category)

    def build(self):
        data = {
            'subcategory': self.subcategories,
            'category': self.categories
        }
        df = pd.DataFrame(data)
        df = df.sort_values('category')
        print(df.to_markdown())
        self.subcategory_to_id = {row["subcategory"]: index for index, (_, row) in enumerate(df.iterrows())}

    def subcat_to_vector(self, subcategory):
        vector = np.zeros(1, dtype=np.float32)
        vector[0] = vector[0] + self.subcategory_to_id[subcategory]
        return vector

    def length(self):
        return len(self.subcategory_to_id)