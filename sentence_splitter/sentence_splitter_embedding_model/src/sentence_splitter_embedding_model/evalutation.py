import numpy as np
import evaluate
from transformers import pipeline
from pathlib import Path

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
    for i, novel_line in enumerate(novel_lines):
        for _ in range(len(novel_line) - 1):
            labels.append(0)
        labels.append(1)
        if i < len(novel_lines) - 1:
            labels.append(0)
    return labels


def sequence_eval(inference_pipeline, sequence_lines):
    sequence = " ".join(sequence_lines)
    prediction = inference_pipeline(sequence)
    prediction_labels = labels_from_prediction(prediction)
    golden_labels = labels_from_novel(sequence_lines)

    if (len(prediction_labels) < len(golden_labels)):
        print("Truncating golden labels. You should use a smaller value for NUM_LINES_FOR_EVAL!")
        golden_labels = golden_labels[:len(prediction_labels)]

    return prediction_labels, golden_labels


def groups(novel_lines, num_lines_for_eval):
    groups = len(novel_lines) / num_lines_for_eval
    if len(novel_lines) % num_lines_for_eval is not 0:
        groups += 1
    return groups


def eval(inference_pipeline, novel_lines, num_lines_for_eval):
    metric = evaluate.load("f1", average="binary")

    grouped_lines = np.array_split(novel_lines, groups(novel_lines, num_lines_for_eval))
    for sequence_lines in grouped_lines:
        prediction_labels, golden_labels = sequence_eval(inference_pipeline, sequence_lines)
        metric.add_batch(predictions=prediction_labels, references=golden_labels)

    return metric.compute()    


def eval_model(model_name: str, novel_path: Path, num_lines_for_eval: int):
    model_checkpoint = "fax4ever/" + model_name
    inference_pipeline = pipeline("token-classification", model=model_checkpoint, aggregation_strategy="simple")
    novel_lines = novel_path.read_text(encoding="utf-8").splitlines()
    return eval(inference_pipeline, novel_lines, num_lines_for_eval)



