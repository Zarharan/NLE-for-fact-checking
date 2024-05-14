import argparse
import torch
import os
import pandas as pd
# import evaluate
import pickle
import warnings
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from dotenv import dotenv_values
from dataset import *
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from pathlib import Path


# take Hugging face and Open AI APIs' secret keys from .env file.
secret_keys = dotenv_values(".env")
HF_TOKEN= secret_keys["HF_TOKEN"]

# metric = evaluate.load("rouge")
warnings.filterwarnings("ignore")


def report_veracity_metrics(target_list, pred_list):
    # print("pred_list len:", len(pred_list), "target_list len:",len(target_list))
    # print("Generated result:", pred_list, "len: ", len(pred_list))

    def _clean_generated_labels(label):
      label= label.strip().replace("#","").replace("\n","").lower().split()
      if len(label)>0:
        return label[0]
      return "other"

    pred_list = list(map(_clean_generated_labels, pred_list))
    target_list = [x.lower().strip() for x in target_list]

    # print("pred_list:", pred_list, "len: ", len(pred_list))
    labels=['true', 'false', 'mixture', 'unproven']

    target_avg= 'weighted'

    metrics_result= {"acc": accuracy_score(y_pred=pred_list, y_true=target_list)
        , "pre_wei": precision_score(y_pred=pred_list, y_true=target_list, average=target_avg, labels=labels)
        , "rec_wei": recall_score(y_pred=pred_list, y_true=target_list, average=target_avg, labels=labels)
        , "f1_wei": f1_score(y_pred=pred_list, y_true=target_list, average=target_avg, labels=labels)
        , "confmat": confusion_matrix(y_pred=pred_list, y_true=target_list, labels=labels)
        , "pre_mac": precision_score(y_pred=pred_list, y_true=target_list, average="macro", labels=labels)
        , "rec_mac": recall_score(y_pred=pred_list, y_true=target_list, average="macro", labels=labels)
        , "f1_mac": f1_score(y_pred=pred_list, y_true=target_list, average="macro", labels=labels)}
    # log("Metrics by using:", target_avg)
    print(metrics_result)

    return metrics_result


def main(args):

    peft_model_id = args.experiment_dir
    # peft_model_id = f"{experiment}/assets"

    # load base LLM model and tokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        token= HF_TOKEN
    )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(peft_model_id, token= HF_TOKEN)

    # test set
    remove_columns= ['claim_id', 'claim', 'explanation', 'summarized_text', 'label']
    test_set= read_dataset(file_path=os.path.dirname(os.getcwd()) + args.data_path, task_type= args.task_type
        , validate_mode= True)
    preprocessed_test_set = preprocess_dataset(tokenizer, args.max_seq_length, args.seed, test_set, remove_columns=remove_columns)

    results = []
    oom_examples = []
    instructions, labels = preprocessed_test_set["text"], preprocessed_test_set["labels"]

    for instruct, label in tqdm(zip(instructions, labels)):
        input_ids = tokenizer(instruct, return_tensors="pt", truncation=True).input_ids.cuda()

        with torch.inference_mode():
            try:
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    # do_sample=True,
                    # top_p=0.95,
                    # temperature=1e-3,
                )
                result = tokenizer.batch_decode(
                    outputs.detach().cpu().numpy(), skip_special_tokens=True
                )[0]
                result = result[len(instruct) :]
            except:
                result = ""
                oom_examples.append(input_ids.shape[-1])

            results.append(result)

    return report_veracity_metrics(labels, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-experiment_dir", default="results/veracity/mistralai/Mistral-7B-v0.1/15_rank-8_dropout-0.1/", type=str)
    parser.add_argument(
        "-max_seq_length",default=830, type=int, help="max sequence length for model and packing of the dataset")
    parser.add_argument(
        "-seed",default=313, type=int, help="seed to replicate results")
    parser.add_argument(
        "-task_type", default="veracity", help = "The target task", choices=['veracity', 'explanation', 'joint'])
    parser.add_argument(
        "-max_new_tokens",default=3, type=int, help="max number of generated tokens")
    parser.add_argument(
        "-data_path",default="/data/pubhealth/summarization/dev_chatgpt_summarized.csv", type=str, help="path of the target file for inference")

    args = parser.parse_args()

    checkpoint_lst= [args.experiment_dir + x for x in os.listdir(args.experiment_dir)]
    save_path= args.experiment_dir
    results = []
    for checkpoint in checkpoint_lst:
        try:
            if not os.path.isdir(checkpoint):
                continue

            print("-"*50, checkpoint)
            args.experiment_dir= checkpoint
            result= main(args)
            result["key"]= checkpoint
            results.append(result)
        except:
            print(f"Error in {checkpoint}")

    #  save results in a csv file
    Path(save_path).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_dict(results)
    df.to_csv(f"{save_path}results.csv")
