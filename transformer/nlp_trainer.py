from transformers import TrainingArguments, Trainer
from nlp_hyper_params import NLPHyperParams, compute_metrics
from nlp_encoder_model import NLPEncoderModel

class NLPTrainer:
    def __init__(self, params: NLPHyperParams, model: NLPEncoderModel, train_dataset, eval_dataset):
        training_args = TrainingArguments(
            output_dir="training_dir",                      # output directory [Mandatory]
            num_train_epochs=params.epochs,                 # total number of training epochs
            per_device_train_batch_size=params.batch_size,  # batch size per device during training
            warmup_steps=500,                               # number of warmup steps for learning rate scheduler
            weight_decay=params.weight_decay,               # strength of weight decay
            save_strategy="no",
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

    def train_and_evaluate(self):
        self.trainer.train()
        print(self.trainer.evaluate())