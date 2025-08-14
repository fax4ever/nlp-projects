# Sentence Splitter

## Models tested

1. 🚀 ModernBERT-base-ita (Most Recent - Dec 2024)
  * `DeepMount00/ModernBERT-base-ita`

2. 🇮🇹 Italian BERT XXL (Most Established): 
  * `dbmdz/bert-base-italian-xxl-cased`

3. 🌍 XLM-RoBERTa (Best Multilingual)
  * `FacebookAI/xlm-roberta-base`
  * `FacebookAI/xlm-roberta-large`

4. 🔬 Italian ELECTRA (Alternative Architecture): 
  * `dbmdz/electra-base-italian-xxl-cased-discriminator`

The project is managed with `uv`.

## Create the uv environment with modules

From the project root (`sentence_splitter/`):
```shell
uv sync --all-packages
```

This creates a virtual environment under `sentence_splitter/.venv`.
Use the `--all-packages` flag to install all subpackages into the shared environment.

To verify the environment has the modules, run from the project root (`sentence_splitter/`):
```shell
uv run main.py
```

## Run tests
```shell
uv run pytest
```

You should see:
```shell
=== 1 passed in 0.01s ===
```

## Use the virtual environment in your IDE

The key idea is to define a single virtual environment (`sentence_splitter/.venv`) at the project root
and install all packages into it. This allows packages to depend on each other and be used from the root.

### IntelliJ IDEA

Open any Python file and click `Configure Python Interpreter`.
Click **Edit** of Module SDK, then **Add new Python SDK**, and select `sentence_splitter/.venv`.

### VSCode / Cursor

Cursor (and likely VSCode) loads the environment by default when you open the project root folder.

### PyTorch with ROCm 6.4.0

If you have an AMD GPU and want to use it with PyTorch, install a specific PyTorch build:

```shell
UV_HTTP_TIMEOUT=3000 uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4/
```

## Other uv commands

### Add a dependency

From the project root (`sentence_splitter/`):
```shell
uv add pandas
```

### Add a new package (module)

From the project root (`sentence_splitter/`):
```shell
uv init --package util
```

### Used Out of Domain dataset

@inproceedings{redaelli-sprugnoli-2024-sentence,
	title = {Is Sentence Splitting a Solved Task? Experiments to the Intersection between {NLP} and {I}talian Linguistics},
	author = {Redaelli, Arianna  and Sprugnoli, Rachele},
	editor = {Dell'Orletta, Felice  and Lenci, Alessandro  and Montemagni, Simonetta  and Sprugnoli, Rachele},
	booktitle = {Proceedings of the 10th Italian Conference on Computational Linguistics (CLiC-it 2024)},
	month = dec,
	year = "2024",
	address = "Pisa, Italy",
	publisher = "CEUR Workshop Proceedings",
	url = "https://aclanthology.org/2024.clicit-1.88/",
	pages = "813--820",
	ISBN = "979-12-210-7060-6",
	abstract = "Sentence splitting, that is the segmentation of the raw input text into sentences, is a fundamental step in text processing. Although it is considered a solved task for texts such as news articles and Wikipedia pages, the performance of systems can vary greatly depending on the text genre. This paper presents the evaluation of the performance of eight sentence splitting tools adopting different approaches (rule-based, supervised, semi-supervised, and unsupervised learning) on Italian 19th-century novels, a genre that has not received sufficient attention so far but which can be an interesting common ground between Natural Language Processing and Digital Humanities."
}