import wikipedia

categories = {
    "Art": ["Leonardo da Vinci", "The Louvre", "Impressionism"],
    "Science": ["Quantum mechanics", "CRISPR", "Large Hadron Collider"],
    "Business": ["Entrepreneurship", "Stock market", "Cryptocurrency"]
}

dataset = []
for label, titles in categories.items():
    for title in titles:
        try:
            summary = wikipedia.summary(title, sentences=2)
            dataset.append((summary, label))
        except Exception as e:
            print(f"Skipping {title}: {e}")
