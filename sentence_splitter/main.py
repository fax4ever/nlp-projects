from sentence_splitter_embedding_model import utils
import torch


def main():
    print("Hello from sentence-splitter!", utils.helper(), torch.cuda.is_available())


if __name__ == "__main__":
    main()
