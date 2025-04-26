from transformers import AutoModelForSequenceClassification

TRANSFORMER_MODELS = ['culturalitems-bigbird', 'culturalitems-roberta-large', 'culturalitems-roberta-base',
                      'culturalitems-roberta-base-5', 'culturalitems-distilbert']

def main():
    for name in TRANSFORMER_MODELS:
        repo = 'fax4ever/' + name
        print("loading", repo)
        model = AutoModelForSequenceClassification.from_pretrained(repo)
        print("loading", model)
        model.save_pretrained(name)
        print("saved", model)

if __name__ == "__main__":
    main()