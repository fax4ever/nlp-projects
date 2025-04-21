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
python3 no_transformer/nlp_app.py
```

## Part 1 - No transformer

Neural networks with 4 inputs:

1. Vector for the description

Frequency vector having the size of the vocabulary of the description

2. Vector for the wikitext

Frequency vector having the size of the vocabulary of the wikitext

3. Vectors for the keys (languages) used by labels, descriptions, aliases, wikipedia_pages

Frequency vectors having the size of the vocabulary of the languages codes.
They will be vector-mapped using the same dictionary.
Since the string are enumerations belonging to same common set.

4. Vector for claims keys

Frequency vector having the size of the vocabulary of the claims (relation types).
Maybe we could manually select a subset of most significant claims. 
At the first attempt let's try to make NN learning this selection (learning parameters).

5. Categorical inputs for category > subcategories

We can map those as numbers bet 0 and the size of all possible values - 1.
Since subcategories determines the category,
we want to order the subcategories so that subcategories of the same category will be contiguous.
In this way with a single number we can denote the two fields in a single shot!
This is the table of the categories / subcategories: [categories.md](generated/categories.md)

6. Boolean input for type

Since we have only two types of type: entity vs concept.
We want to produce a boolean value for each entity to pass to the NN.

### The idea of multi modal NN

The dimensions of the vectors are very different.
So we're going to add pre input layers in which we rescale those.

There are some example of rescaling:

```python
# Text input
self.text_embedding = nn.EmbeddingBag(vocab_size, text_embed_dim, mode='mean')
self.text_proj = nn.Linear(text_embed_dim, common_dim)
```

```python
# Categorical input
self.category_embedding = nn.Embedding(category_vocab_size, 32)
self.category_proj = nn.Linear(32, common_dim)
```

```python
# Numeric input
self.numeric_proj = nn.Linear(num_numeric_features, common_dim)
```

See * [05_multi_modal_nn.py](generated/05_multi_modal_nn.py)

### Potential changes / evolution

Starting from text frequency vectors. See [01_text_class.py](generated/01_text_class.py)
Of course with batch! See [04_batch.py](generated/04_batch.py)

#### Try to use EmbeddingBag

See * [02_text_embedding_multiclass.py](generated/02_text_embedding_multiclass.py)

#### Try to use EmbeddingBag + Glove

See * [03_glove.py](generated/03_glove.py)