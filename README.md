# NLP cultural items classification

1. Set env variable `HUGGINGFACE_TOKEN` to token having read access to contents of all public gated repos you can access.

2. In order to use dumped files, copy:
  * [training.bin](training.bin)
  * [validation.bin](validation.bin)
to the root of the project.
Otherwise, those file will be created for the next time!

3. Optionally, activate the correct environment, in my case is named `pytorch-env`

```bash
conda activate pytorch-env
```

4. Run the project

```bash
python3 nlp_app.py
```

## Part 1 - No transformer

Neural networks with 4 inputs:

1. Vector for the description

Binary vector having the size of the vocabulary of the description

2. Vector for the wikitext

Binary vector having the size of the vocabulary of the wikitext

3. Vector for the languages used by labels, descriptions, aliases, wikipedia_pages

4 Binary vectors concatenated (?) or weighted, each having the size of the vocabulary of the language used

4. Vector for relevant claims (filtering the claims that may be representative) and the subcategory

1 binary vector containing the relevant claims + 1 binary vector subcategory

Each of this vector is rescaled to an input vector having 64 dimensions.
So the first layer just rescales each input without mixing them.

After that we have 2 hidden layers and 1 output layer with 3 output features.

This is what is called a multi-modal neural network.

Verify if it is best to use embedding bags / glove embedding bags instead of simple binary vectors based on vocabulary.