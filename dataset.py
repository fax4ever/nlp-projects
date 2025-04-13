import os, requests
from wikidata.client import Client
from datasets import load_dataset
from entity import Entity

API_URL = "https://en.wikipedia.org/w/api.php"

def extract_entity_id(url):
    return url.strip().split("/")[-1]

class DataAccess:
    def __init__(self):
        self.dataset = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset', token=os.environ['HUGGINGFACE_TOKEN'])
        self.client = Client()

    def train(self, index):
        return self.dataset['train'][index]

    def entity(self, entity_id):
        return self.client.get(entity_id, load=True)

    def wiki_text(self, enwiki):
        if not enwiki:
            return None
        title = enwiki["title"]
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": True,
            "titles": title,
            "format": "json",
            "redirects": 1
        }
        res = requests.get(API_URL, params=params).json()
        page = next(iter(res["query"]["pages"].values())) #see below the dictionary, we create an iterable and pick the first and only item
        extract = page.get("extract", "")
        return " ".join(extract.split())[:1000].lower()  # we get the "extract field"

    def feature(self, index):
        train = self.train(index)
        entity_id = extract_entity_id(train['item'])
        entity = self.entity(entity_id)
        sitelinks = entity.data.get("sitelinks", {})
        enwiki = sitelinks.get("enwiki")
        return Entity(entity_id, train, entity, self.wiki_text(enwiki))




