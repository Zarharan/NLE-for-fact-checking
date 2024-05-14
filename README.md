# NLE-for-fact-checking

**This repository will be updated soon.**

## This repo is for our TrustNLP 2024 paper "Tell Me Why: Explainable Public Health Fact-Checking with Large Language Models".

In this repository, we explored Natural Language Explanation (NLE) for fact-checking. We used [PubHealth](https://github.com/neemakot/Health-Fact-Checking) dataset and you can find the dataset and our saveed prompts templates in the [data](https://github.com/Zarharan/NLE-for-fact-checking/tree/main/data) directory. We utilized prompt-based paradigm to generate explanation by using GPT-3 and ChatGPT for both zero-shot and few-shot learning.

## Reproducing the Experiments

In order to reproduce the results of our experiments, you can run ``python PubHealth_experiments.py`` with various arguments. To cite an example, by running the following command, you can reproduce our zero-shot result by using GPT-3 and create an appropriate prompt for querying ChatGPT.

```
python PubHealth_experiments.py -summarize gpt3 -k_per_class 4 -k_rand_instance 4 -test_path data/pubhealth/test.tsv -add_chatgpt_prompt True
```

You can find the description of each argument in [PubHealth_experiments.py file](https://github.com/Zarharan/NLE-for-fact-checking/blob/main/PubHealth_experiments.py)
