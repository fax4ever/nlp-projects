from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import DatasetDict
import evaluate
import os
import numpy as np

END_OF_SENTENCE = 1
NOT_END_OF_SENTENCE = 0
LABEL_FOR_START_END_OF_SEQUENCE = NOT_END_OF_SENTENCE

class SplitterWithEncoder:

    def tokenize_dataset(self, dataset_dict:DatasetDict, embedding_model:str):
        self.embedding_model = embedding_model
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.tokenized_dataset_dict = dataset_dict.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset_dict["train"].column_names,
            batch_size=128,
        )


    def train(self):
        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)
        self.metric = evaluate.load("seqeval")
        self.model = AutoModelForTokenClassification.from_pretrained(self.embedding_model, num_labels=2)
        args = TrainingArguments(
            "bert-base-cased-sentence-splitter",
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
            train_dataset=self.tokenized_dataset_dict["train"],
            eval_dataset=self.tokenized_dataset_dict["validation"],
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            processing_class=self.tokenizer,
        )
        trainer.train()
        trainer.push_to_hub(commit_message="Training complete", token=os.environ['HF_TOKEN'])


    def tokenize_and_align_labels(self, items):
        tokenized_inputs = self.tokenizer(
            items["tokens"], is_split_into_words=True
        )

        all_labels = items["labels"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs


    def align_labels_with_tokens(self, labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = LABEL_FOR_START_END_OF_SEQUENCE if word_id is None else labels[word_id]
                new_labels.append(label)
            else:
                # Treat the same word never as end of sentence
                new_labels.append(NOT_END_OF_SENTENCE)
        return new_labels
    

    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        all_metrics = self.metric.compute(predictions=predictions, references=labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }
