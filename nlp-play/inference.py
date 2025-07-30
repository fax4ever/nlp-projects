from token_classification.token_classification_inference import TokenClassificationInference


def main():
    service = TokenClassificationInference()
    result = service.token_classify("I only got myself And this big ol'")
    print(result)


if __name__ == "__main__":
    main()
