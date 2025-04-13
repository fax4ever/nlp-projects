from SPARQLWrapper import SPARQLWrapper, JSON

def query_wikidata(limit=100):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(f"""
    SELECT ?item ?itemLabel ?description WHERE {{
      ?item wdt:P31 wd:Q5.
      ?item schema:description ?description.
      FILTER(LANG(?description) = "en")
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit}
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    dataset = []
    for result in results["results"]["bindings"]:
        description = result["description"]["value"]
        # Assign labels here, e.g., via keyword or heuristics
        if "painter" in description or "artist" in description:
            label = "Art"
        elif "scientist" in description:
            label = "Science"
        elif "business" in description:
            label = "Business"
        else:
            continue
        dataset.append((description, label))
    return dataset
