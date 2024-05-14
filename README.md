# NLE-for-fact-checking

**This repository will be updated soon.**

## This repo is for our TrustNLP 2024 paper "Tell Me Why: Explainable Public Health Fact-Checking with Large Language Models".

In this repository, we explored Natural Language Explanation (NLE) for fact-checking. We used [PubHealth](https://github.com/neemakot/Health-Fact-Checking) dataset and you can find the dataset and our saveed prompts templates in the [data](https://github.com/Zarharan/NLE-for-fact-checking/tree/main/data) directory. We utilized prompt-based paradigm to generate explanation by using closed-source LLMs including GPT-3 (text-davinci-003), ChatGPT (gpt-3.5-turbo), and GPT-4 and  for both zero-shot and few-shot learning. We also used publicly available LLMs including Falcon-180B, Llama-70B, Vicuna-13B, and Mistral-7B for zero- and few-shot senario. We implemented PEFT with Vicuna-13B and Mistral-7B for veracity prediction, explanation generation, and the joint setting.

## Reproducing the Experiments

In order to reproduce the results of our experiments, you can run ``python PubHealth_experiments.py`` with various arguments. To cite an example, by running the following command, you can reproduce our zero-shot result of the veracity prediction task on the test set by using GPT-4.

```
python3 PubHealth_experiments.py -test_path data/pubhealth/test.tsv -summarize chatgpt -summarization_max_token 350 -prompt_template veracity/claude_suggestion -explanation_max_token 3 -test_target_set test -k_per_class 0 -prompt_type zero -plm gpt4 -plm_engine gpt-4 -nle_temperature 1.0 -k_rand_instance 1233

```

You can find the description of each argument in [PubHealth_experiments.py file](https://github.com/Zarharan/NLE-for-fact-checking/blob/main/PubHealth_experiments.py)
