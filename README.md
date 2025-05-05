# NLP cultural items classification

Base Dataset: https://huggingface.co/datasets/sapienzanlp/nlp2025_hw1_cultural_dataset
The task is to classify each item into one of the three categories:
* Cultural Agnostic (CA)
* Cultural Representative (CR)
* Cultural Exclusive (CE)

The project involves two distinct approaches:

* Non-LM-based: Transformer models are not allowed.
* LM-based: Only encoder-based transformer models may be used.

## Content

1. [Non-LM-based part: Training notebook](colabs/NLP_no_transformer_training.ipynb)
2. [Non-LM-based part: Inference notebook](colabs/NLP_no_transformer_inference.ipynb)
3. [LM-based part: Training notebook](colabs/NLP_yes_transformer_training.ipynb)
4. [LM-based part: Inference notebook](colabs/NLP_yes_transformer_inference.ipynb)
5. [Report](latex/nlp-homework-1.pdf)
6. [Python code for the Non-LM-based part](no_transformer)
7. [Python code for the LM-based part](transformer)
8. [Test set - unlabeled](test_unlabeled.csv)
