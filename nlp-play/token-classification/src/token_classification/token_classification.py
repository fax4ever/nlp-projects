from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import os

def align_labels_with_tokens(labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels

class TokenClassification:
    def __init__(self):
        model_checkpoint = "bert-base-cased"
        self.dataset = load_dataset("conll2003", trust_remote_code=True)
        ner_feature = self.dataset["train"].features["ner_tags"]
        self.label_names = ner_feature.feature.names
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        print("tokenizer.is_fast", self.tokenizer.is_fast)
        self.tokenized_dataset = self.dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=self.dataset["train"].column_names,
        )
        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)
        self.metric = evaluate.load("seqeval")
        self.id2label = {i: label for i, label in enumerate(self.label_names)}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        print("num_labels", self.model.config.num_labels)

    def train(self):    
        args = TrainingArguments(
            "bert-finetuned-ner",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=3,
            weight_decay=0.01,
            push_to_hub=True,
            hub_token=os.environ['HF_TOKEN']
        )
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            processing_class=self.tokenizer,
        )
        trainer.train()
        trainer.push_to_hub(commit_message="Training complete", token=os.environ['HF_TOKEN'])

    def tokenize_and_align_labels(self, items):
        tokenized_inputs = self.tokenizer(
            items["tokens"], truncation=True, is_split_into_words=True
        )
        all_labels = items["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs
    
    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[self.label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = self.metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }

    
