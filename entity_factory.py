import requests
from entity import Entity

API_URL = "https://en.wikipedia.org/w/api.php"

def extract_entity_id(url):
    return url.strip().split("/")[-1]

def wiki_text(en_wiki):
    if not en_wiki:
        return None
    title = en_wiki["title"]
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
    # we get the "extract field"
    # removing new lines and extra whitespaces
    # lowercasing it - taking the first 1000 elements
    # usually the most representative
    return " ".join(extract.split())[:1000].lower()

class EntityFactory:
    def __init__(self, client):
        self.client = client

    def create(self, item):
        entity_id = extract_entity_id(item['item'])
        wikidata = self.client.get(entity_id, load=True)
        sitelinks = wikidata.data.get("sitelinks", {})
        en_wiki = sitelinks.get("enwiki")
        return Entity(entity_id, item, wikidata, wiki_text(en_wiki))