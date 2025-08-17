from datasets import load_dataset


SIZE = 192 # Number of words to put on each input of the encoder model


def words_to_sentences(words):
    input_text = " ".join(words)
    input_text = input_text.replace(" ,", ",")
    input_text = input_text.replace(" .", ".")
    input_text = input_text.replace(" ?", "?")
    input_text = input_text.replace(" !", "!")
    input_text = input_text.replace(" :", ":")
    input_text = input_text.replace(" ;", ";")
    input_text = input_text.replace("' ", "'")
    return input_text


def create_conversations(examples):
    input_texts = []
    output_texts = []

    for tokens, labels in zip(examples['tokens'], examples['labels']):
        input_text = words_to_sentences(tokens)
        input_texts.append(input_text)

        sentences = []
        current_sentence = []
        for token, label in zip(tokens, labels):
            current_sentence.append(token)
            if label == 1:  # End of sentence
                sentences.append(words_to_sentences(current_sentence))
                current_sentence = []

        if current_sentence:
            sentences.append(words_to_sentences(current_sentence))

        output_text = "\n".join([f"{i+1}. {sentence}" for i, sentence in enumerate(sentences)])
        output_texts.append(output_text)

    return {"input_text" : input_texts, "output_text" : output_texts}


def conversations_dataset():
    dataset_dict = load_dataset(f"fax4ever/manzoni-{SIZE}")
    return dataset_dict.map(create_conversations, batched = True)