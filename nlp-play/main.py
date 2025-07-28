from token_classification.token_classification import TokenClassification


def main():
    service = TokenClassification('Frank')
    print(service.hello())


if __name__ == "__main__":
    main()
