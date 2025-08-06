import pandas as pd
from datasets import Dataset

class Prompt:
    def __init__(self, input_text):
        self.input_text = input_text

    def instruction(self):
        return f"""Dividi il seguente testo italiano in frasi. Per favore rispondi con una frase per riga. Grazie.

Testo: {self.input_text}
"""

    def conversation(self, output_text):
        return[
            {"role" : "system",    "content" : "Sei un esperto di linguistica italiana specializzato nella segmentazione delle frasi."},
            {"role" : "user",      "content" : self.instruction()},
            {"role" : "assistant", "content" : output_text},
        ]

    def question(self):
        return[
            {"role" : "system",    "content" : "Sei un esperto di linguistica italiana specializzato nella segmentazione delle frasi."},
            {"role" : "user",      "content" : self.instruction()},
        ]


def create_conversations(examples):
    input_texts  = examples["input_text"]
    output_texts = examples["output_text"]

    conversations = []
    for input_text, output_text in zip(input_texts, output_texts):
        conversations.append(Prompt(input_text).conversation(output_text))

    return { "conversations": conversations, }


def create_question(input_text):
    return Prompt(input_text).question()


class ConversationFactory:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def llm_dataset(self, dataset):
        data = pd.Series(self.create_conversations(dataset))
        data.name = "text"
        result = Dataset.from_pandas(pd.DataFrame(data))
        result = result.shuffle(seed=3407)
        return result

    def create_conversations(self, dataset):
        return self.apply_chat_template(
            dataset.map(create_conversations, batched = True)["conversations"],
        )

    def llm_question(self, input_text):
        return self.apply_chat_template(
            [create_question(input_text)],
        )

    def apply_chat_template(self, conversations):
        return self.tokenizer.apply_chat_template(
            conversations,
            tokenize = False,
        )
