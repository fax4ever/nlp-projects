# Sentence Splitter

## Model tested

1. 🚀 ModernBERT-base-ita (Most Recent - Dec 2024)
  * `DeepMount00/ModernBERT-base-ita`

2. 🇮🇹 Italian BERT XXL (Most Established): 
  * `dbmdz/bert-base-italian-xxl-cased`

3. 🌍 XLM-RoBERTa (Best Multilingual)
  * `FacebookAI/xlm-roberta-base`
  * `FacebookAI/xlm-roberta-large`

4. 🔬 Italian ELECTRA (Alternative Architecture): 
  * `dbmdz/electra-base-italian-xxl-cased-discriminator`

The project is managed using UV.

## Create the UV environment with Modules

On the root (sentence_splitter)
```shell
uv sync --all-packages
```

After this command a virtual environment will be created under `sentence_splitter/.venv`
But the modules are not loaded into it without the option `--all-packages`!

To test that the modules are present by the virtual environment:

On the root (sentence_splitter)
```shell
uv run main.py
```

## Execute the tests

Run the tests
```shell
uv run pytest
```

You should see:
```shell
=== 1 passed in 0.01s ===
```

## Install the virtual environment on your IDE

The crucial idea of UV packaging is to define a single virtual environment (`sentence_splitter/.venv`)
at root project level **and then** (it means...) install all the packages on it!
In this way you can make one package using another one... or use packages from the root of the project.
The common virtual environment will be common denominator!

### IntelliJ IDEA

To install it on IntelliJ IDEA open any Python file and click on `Configure Python Interpreter`.
Then click on the **Edit** of ModuleSDK.
Then click on **Add new Python SDK**
Select the virtual environment: `sentence_splitter/.venv`!

### VSCode / Cursor

With Cursor (and maybe VSCode -- I didn't try it) the environment is loaded by default if you open the project
with the root project directory as target folder.

### ROCm 6.4.0 PyTorch 

If you have an AMD Radeon and you want to use it with Pytorch, you can install a specific Pytorch version:

```shell
UV_HTTP_TIMEOUT=3000 uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4/
```

## UV other commands

### Add a dependency

On the root (sentence_splitter)
```shell
uv add pandas
```

### Add a new package (module) 

On the root (sentence_splitter)
```shell
uv init --package util
```