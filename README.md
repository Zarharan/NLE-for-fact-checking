# NLE-for-fact-checking

## This repo is for our TrustNLP 2024 paper "[Tell Me Why: Explainable Public Health Fact-Checking with Large Language Models](https://aclanthology.org/2024.trustnlp-1.21)".

In this repository, we explored Natural Language Explanation (NLE) for fact-checking. We used [PubHealth](https://github.com/neemakot/Health-Fact-Checking) dataset and you can find the dataset and our saveed prompts templates in the [data](https://github.com/Zarharan/NLE-for-fact-checking/tree/main/data) directory. We utilized prompt-based paradigm to generate explanation by using closed-source LLMs including GPT-3 (text-davinci-003), ChatGPT (gpt-3.5-turbo), and GPT-4 and  for both zero-shot and few-shot learning. We also used publicly available LLMs including Falcon-180B, Llama-70B, Vicuna-13B, and Mistral-7B for zero- and few-shot senario. We implemented PEFT with Vicuna-13B and Mistral-7B for veracity prediction, explanation generation, and the joint setting.

## Reproducing the Experiments

In order to reproduce the results of our experiments, you can run ``python PubHealth_experiments.py`` with various arguments. To cite an example, by running the following command, you can reproduce our zero-shot result of the veracity prediction task on the test set by using GPT-4.

```
python3 PubHealth_experiments.py -test_path data/pubhealth/test.tsv -summarize chatgpt -summarization_max_token 350 -prompt_template veracity/claude_suggestion -explanation_max_token 3 -test_target_set test -k_per_class 0 -prompt_type zero -plm gpt4 -plm_engine gpt-4 -nle_temperature 1.0 -k_rand_instance 1233
```

You can find the description of each argument in [PubHealth_experiments.py file](https://github.com/Zarharan/NLE-for-fact-checking/blob/main/PubHealth_experiments.py)

## Human Evaluation Tool

You can locate the human evaluation tool for assessing explanations based on the proposed criteria in the paper within [this repository](https://github.com/Zarharan/human-evaluation-tool-for-NLE).

## Reference

If you use the tool or any information from this repository or the paper, please cite the paper using the format provided below.

```
@inproceedings{zarharan-etal-2024-tell,
    title = "Tell Me Why: Explainable Public Health Fact-Checking with Large Language Models",
    author = "Zarharan, Majid  and
      Wullschleger, Pascal  and
      Behkam Kia, Babak  and
      Pilehvar, Mohammad Taher  and
      Foster, Jennifer",
    editor = "Ovalle, Anaelia  and
      Chang, Kai-Wei  and
      Cao, Yang Trista  and
      Mehrabi, Ninareh  and
      Zhao, Jieyu  and
      Galstyan, Aram  and
      Dhamala, Jwala  and
      Kumar, Anoop  and
      Gupta, Rahul",
    booktitle = "Proceedings of the 4th Workshop on Trustworthy Natural Language Processing (TrustNLP 2024)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.trustnlp-1.21",
    doi = "10.18653/v1/2024.trustnlp-1.21",
    pages = "252--278",
    abstract = "This paper presents a comprehensive analysis of explainable fact-checking through a series of experiments, focusing on the ability of large language models to verify public health claims and provide explanations or justifications for their veracity assessments. We examine the effectiveness of zero/few-shot prompting and parameter-efficient fine-tuning across various open and closed-source models, examining their performance in both isolated and joint tasks of veracity prediction and explanation generation. Importantly, we employ a dual evaluation approach comprising previously established automatic metrics and a novel set of criteria through human evaluation. Our automatic evaluation indicates that, within the zero-shot scenario, GPT-4 emerges as the standout performer, but in few-shot and parameter-efficient fine-tuning contexts, open-source models demonstrate their capacity to not only bridge the performance gap but, in some instances, surpass GPT-4. Human evaluation reveals yet more nuance as well as indicating potential problems with the gold explanations.",
}
```
