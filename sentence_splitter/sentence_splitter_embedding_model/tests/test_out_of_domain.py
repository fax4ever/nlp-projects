from sentence_splitter_embedding_model.evalutation import eval_model
from pathlib import Path


def test_eval_model():
    value = eval_model("bert-base-cased-sentence-splitter", Path(__file__).parent / "Cuore-GOLD.txt", 6)
    print(value)
    print(value["f1"])