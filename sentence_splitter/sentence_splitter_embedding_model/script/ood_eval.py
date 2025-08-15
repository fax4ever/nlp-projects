from sentence_splitter_embedding_model.evalutation  import eval_model
from pathlib import Path


if __name__ == "__main__":

    models = [
        "bert-base-cased-sentence-splitter",
        "ModernBERT-base-ita-sentence-splitter",
        "bert-base-italian-xxl-cased-sentence-splitter",
        "xlm-roberta-base-sentence-splitter",
        "xlm-roberta-large-sentence-splitter",
        "electra-base-italian-xxl-cased-discriminator-sentence-splitter",
    ]

    for model_name in models:
        novel = "Cuore-GOLD.txt"
        value = eval_model(model_name, Path(__file__).parent / novel, 6)
        print(model_name, novel, value["f1"])

        novel = "Malavoglia-GOLD.txt"
        value = eval_model(model_name, Path(__file__).parent / novel, 3)
        print(model_name, novel, value["f1"])

        novel = "Pinocchio-GOLD.txt"
        value = eval_model(model_name, Path(__file__).parent / novel, 7)
        print(model_name, novel, value["f1"])

        novel = "Quarantana-GOLD.txt"
        value = eval_model(model_name, Path(__file__).parent / novel, 5)
        print(model_name, novel, value["f1"])