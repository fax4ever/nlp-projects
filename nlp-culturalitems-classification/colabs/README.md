# NLP Homework 1 - File description

## Colab(s):

1. [NLP_no_transformer_training.ipynb](NLP_no_transformer_training.ipynb): non-LM-based training Colab, the trained model is pushed to https://huggingface.co/fax4ever/culturalitems-no-transformer
2. [NLP_no_transformer_inference.ipynb](NLP_no_transformer_inference.ipynb): non-LM-based inference Colab, the trained model is pulled from the same repo
3. [NLP_yes_transformer_training.ipynb](NLP_yes_transformer_training.ipynb): LM-based training Colab, the trained model is pushed to https://huggingface.co/fax4ever/culturalitems-roberta-base
4. [NLP_yes_transformer_inference.ipynb](NLP_yes_transformer_inference.ipynb): LM-based inference Colab, the trained model is pulled from the same repo

## Report:

5. [nlp-homework-1.pdf](nlp-homework-1.pdf)

## Inference of the test set:

6. Lost_in_Language_Recognition_output_multimodalnn.csv: non-LM-based
7. Lost_in_Language_Recognition_output_roberta.csv: LM-based

## Dump files

Note: those files are not mandatory. If those file are not present the Colab(s) will 
recreate them, so that they can be used for (loaded by) the next run.

### Data loaded dumps

These dumps contain data retrieved from Wikipedia pages and Wikidata entities online. 
Since this content is mutable, results may vary if the files are regenerated. 
If you want to reproduce the exact results described in the report, please use these files. 
Remove them if your goal is to test the Colab’s ability to generate them from scratch.

8. [training.bin](training.bin)
9. [validation.bin](validation.bin)
10. [test.bin](test.bin)

### Processed data dumps

These are used only by the non-LM-based components. Unlike the others, 
they can be safely deleted without affecting the results, 
as they simply store preprocessed data derived from the original sources. 
Use them to speed up training and inference. 
Remove them if your goal is to test Colab’s ability to recreate them.

11. [training-proc.bin](training-proc.bin)
12. [validation-proc.bin](validation-proc.bin)
13. [test-proc.bin](test-proc.bin)
