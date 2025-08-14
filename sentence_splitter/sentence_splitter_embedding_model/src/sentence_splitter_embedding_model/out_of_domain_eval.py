from transformers import pipeline
from pathlib import Path
import evaluate


def labels_from_prediction(prediction):
    ones = {}
    for label in prediction:
        if label["entity_group"] == "LABEL_1":
            for i in range(label["start"], label["end"]):
                ones[i] = 1

    first = prediction[0]
    last = prediction[-1]

    labels = []
    for i in range(first["start"], last["end"]):
        if i in ones:
            labels.append(1)
        else:
            labels.append(0)
    return labels


def labels_from_novel(novel_lines):
    labels = []
    for _, novel_line in enumerate(novel_lines):
        for i in range(len(novel_line) - 1):
            labels.append(0)
        labels.append(1)
        labels.append(0)
    return labels


class OutOfDomainEval:
    def __init__(self, novel_lines, prediction):
        assert novel_lines is not None
        assert prediction is not None

        prediction_labels = labels_from_prediction(prediction)
        golden_labels = labels_from_novel(novel_lines)
        print(prediction_labels)
        print(golden_labels)

        golden_labels_truncated = golden_labels[:len(prediction_labels)]

        metric = evaluate.load("f1", average="binary")
        metric.add_batch(predictions=prediction_labels, references=golden_labels_truncated)
        f1 = metric.compute()
        print(f1)


def test_out_of_domain_eval():
    trained_model_name = "bert-base-cased-sentence-splitter"
    model_checkpoint = "fax4ever/" + trained_model_name
    inference_pipeline = pipeline("token-classification", model=model_checkpoint, aggregation_strategy="simple")

    cuore_path = Path(__file__).parent / "Cuore-GOLD.txt"
    cuore_lines = cuore_path.read_text(encoding="utf-8").splitlines()
    coure = " ".join(cuore_lines)
    prediction = inference_pipeline(coure)
    OutOfDomainEval(cuore_lines, prediction)
    

if __name__ == "__main__":
    test_out_of_domain_eval()
    