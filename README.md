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