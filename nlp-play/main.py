from token_classification.token_classification import TokenClassification


def main():
    service = TokenClassification()
    print(service.tokenized_dataset)


if __name__ == "__main__":
    main()
