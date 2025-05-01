def wikipedia_pages(sitelinks):
    result = []
    for site_key in sitelinks.keys():
        if site_key.endswith("wiki") and not site_key.startswith("commons"):
            lang = site_key.replace("wiki", "")
            result.append(lang)
    return result

def build_claims(claims):
    result = {}
    for prop_id, values in claims.items():
        result[prop_id] = len(values)
    return result

class Entity:
    def __init__(self, entity_id, dataset_item, wiki_data, wiki_text):
        self.entity_id = entity_id
        if 'label' in dataset_item:
            self.label = dataset_item['label']
        self.name = dataset_item['name']
        self.description = dataset_item['description']
        self.type = dataset_item['type']
        self.category = dataset_item['category']
        self.subcategory = dataset_item['subcategory']
        self.wiki_text = wiki_text
        # Languages
        self.labels = list(wiki_data.data.get("labels", {}).keys())
        self.descriptions = list(wiki_data.data.get("descriptions", {}).keys())
        self.aliases = list(wiki_data.data.get("aliases", {}).keys())
        self.wikipedia_pages = wikipedia_pages(wiki_data.data.get("sitelinks", {}))
        # Properties
        self.claims = build_claims(wiki_data.data.get("claims", {}))

    def __str__(self):
        return self.entity_id + ": " + self.label + " - " + self.name

