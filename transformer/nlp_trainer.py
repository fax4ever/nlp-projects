from copy import deepcopy

from transformers import TrainingArguments, Trainer, TrainerCallback
from nlp_hyper_params import NLPHyperParams, compute_metrics
from nlp_encoder_model import NLPEncoderModel

class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        self._trainer = trainer
        self.train_loss = []
        self.train_accuracy = []
        self.valid_loss = []
        self.valid_accuracy = []

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            train_evaluation = self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            self.train_loss.append(train_evaluation['train_loss'])
            self.train_accuracy.append(train_evaluation['train_accuracy'])
            valid_evaluation = self._trainer.evaluate(eval_dataset=self._trainer.eval_dataset, metric_key_prefix="valid")
            self.valid_loss.append(valid_evaluation['valid_loss'])
            self.valid_accuracy.append(valid_evaluation['valid_accuracy'])
            return control_copy

class NLPTrainer:
    def __init__(self, params: NLPHyperParams, model: NLPEncoderModel, train_dataset, eval_dataset):
        training_args = TrainingArguments(
            output_dir="training_dir",                      # output directory [Mandatory]
            num_train_epochs=params.epochs,                 # total number of training epochs
            per_device_train_batch_size=params.batch_size,  # batch size per device during training
            warmup_steps=500,                               # number of warmup steps for learning rate scheduler
            weight_decay=params.weight_decay,               # strength of weight decay
            save_strategy="no",
            eval_strategy="epoch",
            learning_rate=params.learning_rate,             # learning rate
            report_to="none",
            logging_dir="cultural_analysis_logs"           # use it later to get the training curves
        )
        self.trainer = Trainer(
            model=model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=model.tokenizer,
            data_collator=model.data_collator,
            compute_metrics=compute_metrics,
        )
        self.callback = CustomCallback(self.trainer)
        self.trainer.add_callback(self.callback)

    def train_and_evaluate(self):
        self.trainer.train()
        return {
            "train_loss": self.callback.train_loss,
            "train_accuracy": self.callback.train_accuracy,
            "valid_loss": self.callback.valid_loss,
            "valid_accuracy": self.callback.valid_accuracy
        }