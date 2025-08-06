from datasets import load_dataset
import os


SIZE = 192


def create_conversations(examples):
    input_texts = []
    output_texts = []

    for tokens, labels in zip(examples['tokens'], examples['labels']):
        input_text = " ".join(tokens)
        input_texts.append(input_text)

        sentences = []
        current_sentence = []
        for token, label in zip(tokens, labels):
            current_sentence.append(token)
            if label == 1:  # End of sentence
                sentences.append(" ".join(current_sentence).strip())
                current_sentence = []
        
        # Add remaining tokens if any
        if current_sentence:
            sentences.append(" ".join(current_sentence).strip())

        output_text = "\n".join([f"{i+1}. {sentence}" for i, sentence in enumerate(sentences)])
        output_texts.append(output_text)

    return {"input_text" : input_texts, "output_text" : output_texts}


def main():
    dataset_dict = load_dataset(f"fax4ever/manzoni-{SIZE}")
    llm_dataset_dict = dataset_dict.map(create_conversations, batched = True)
    llm_dataset_dict.push_to_hub(f"fax4ever/llm-manzoni-{SIZE}", token=os.getenv("HF_TOKEN"))


if __name__ == "__main__":
    main()