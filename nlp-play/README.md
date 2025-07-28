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

## Synch the project

To acquire any changes (for instance if a new package is defined or the [pyproject.toml](pyproject.toml) is changed manually):

On the root (nlp-play)
```shell
uv sync --all-packages
```

The main idea of uv is to have a single virtual environment and install it:
1. all the dependencies
2. all the packages

## Container build and run

Build, tag so that the image will be in the Podman local repo:

```shell
podman build -t quay.io/fercoli/nlp-play:0.1.0 .
```

Run the container:

```shell
podman run --rm quay.io/fercoli/nlp-play:0.1.0
```

Replace the entrypoint `uv run main.py` with the `/bin/bash` and run the container,
to inspect the scaffold and debug for instance.

```shell
podman run --rm --name=nlp-play -it quay.io/fercoli/nlp-play:0.1.0 /bin/bash
```

In any case the image can be removed

```shell
podman rmi quay.io/fercoli/nlp-play:0.1.0
```

Or pushed to the remote quay.io

```shell
podman push quay.io/fercoli/nlp-play:0.1.0
```