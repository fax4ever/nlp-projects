from matplotlib import pyplot as plt

from nlp_hyper_params import NLPHyperParams, compute_metrics
from nlp_encoder_model import NLPEncoderModel
from nlp_trainer import NLPTrainer
from wiki_dataset import WikiDataset

def main():
    params = NLPHyperParams()
    dataset = WikiDataset()
    model = NLPEncoderModel(params)

    print("Tokenize the dataset ...")
    tokenized_datasets = dataset.tokenize(model.tokenizer)
    print(tokenized_datasets)

    train_ = tokenized_datasets["train"]
    validation_ = tokenized_datasets["validation"]
    trainer = NLPTrainer(params, model, train_, validation_)
    history = trainer.train_and_evaluate()

    plt.title("MSE Loss - Plot")
    plt.plot(history["train_loss"], label="training loss")
    plt.plot(history["valid_loss"], label="validation loss")
    plt.legend()
    plt.show()

    plt.title("Accuracy - Plot")
    plt.plot(history["train_accuracy"], label="training accuracy")
    plt.plot(history["valid_accuracy"], label="validation accuracy")
    plt.legend()
    plt.show()

    # uncomment to push
    # model.push("fax4ever/culturalitems-transformer")

if __name__ == "__main__":
    main()