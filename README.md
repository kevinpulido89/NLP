# NLP
NLP related code

## MeLi_Challenge
Challenge based on the Mercadolibre dataset published on 2019-2

Visit https://ml-challenge.mercadolibre.com/ for more details.

Solutions availiable here:
- Fully Connected Neural Networks (FastText- SIF - 3 ANN) Performance of 79% on accuracy.
- Recurrent Neural Networks (keras embedding, lstm, bi-lstm, gru, bi-gru) Performance of 81% on accuracy.

*Keywords:* keras, fcnn, rnn, lstm, gru, leakyrelu, selu, relu, , gelu, fasttext, sif, mercadolibre, bilstm, bigru


## BETO para resumen de texto en Español (BERT-like approach for text summmarization)

Ejemplos de clasificación con modelo(s) BERT y BERT-like (BETO)
<!--
## Adversarial evaluation of model performances

Here is an example on evaluating a model using adversarial evaluation of natural language inference with the Heuristic Analysis for NLI Systems (HANS) dataset [McCoy et al., 2019](https://arxiv.org/abs/1902.01007). The example was gracefully provided by [Nafise Sadat Moosavi](https://github.com/ns-moosavi).

The HANS dataset can be downloaded from [this location](https://github.com/tommccoy1/hans).

This is an example of using test_hans.py:

```bash
export HANS_DIR=path-to-hans
export MODEL_TYPE=type-of-the-model-e.g.-bert-roberta-xlnet-etc
export MODEL_PATH=path-to-the-model-directory-that-is-trained-on-NLI-e.g.-by-using-run_glue.py

python examples/hans/test_hans.py \
        --task_name hans \
        --model_type $MODEL_TYPE \
        --do_eval \
        --data_dir $HANS_DIR \
        --model_name_or_path $MODEL_PATH \
        --max_seq_length 128 \
        --output_dir $MODEL_PATH \
```

This will create the hans_predictions.txt file in MODEL_PATH, which can then be evaluated using hans/evaluate_heur_output.py from the HANS dataset.

The results of the BERT-base model that is trained on MNLI using batch size 8 and the random seed 42 on the HANS dataset is as follows:

```bash
Heuristic entailed results:
lexical_overlap: 0.9702
subsequence: 0.9942
constituent: 0.9962

Heuristic non-entailed results:
lexical_overlap: 0.199
subsequence: 0.0396
constituent: 0.118
```
-->
