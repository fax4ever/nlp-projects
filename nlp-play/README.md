# NLP Play

To create a new root project

From outside the project (nlp-projects)
```shell
uv init nlp-play
```

Go to the project directory (nlp-play)

Run the main

On the root (nlp-play)
```shell
uv run main.py
```

Let's add some dependencies:

```shell
uv add transformers
```

This is a test dependency:

```shell
uv add --dev pytest
```

Let's create a module

```shell
uv init --package token-classification
```

Look at the scaffold and see that the module name is `token-classification` 
and this is automatically converted to `token_classification` as package.

## Execute the tests

Go to the module directory (token-classification)
```shell
cd token-classification
```

```shell
mkdir tests
```

Add a test

This can be run from both the module directory (token-classification)
and the root (nlp-play) root directory.
```shell
uv run pytest
```