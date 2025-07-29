from token_classification.token_classification import TokenClassification


def main():
    service = TokenClassification()
    print(service.tokenized_dataset)
    service.train()


if __name__ == "__main__":
    main()
