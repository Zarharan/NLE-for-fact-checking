import os
from random import randrange
from functools import partial
import torch
from torch import nn
from datasets import load_dataset
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          HfArgumentParser,
                          Trainer,
                          TrainingArguments,
                          DataCollatorForLanguageModeling,
                          pipeline,
                          logging,
                          set_seed)
from transformers import EarlyStoppingCallback, IntervalStrategy
import bitsandbytes as bnb
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from trl import SFTTrainer
from datasets import Dataset
import pandas as pd
from dotenv import dotenv_values
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import pickle
import numpy as np
from dataset import *
from pynvml import *


# take Hugging face and Open AI APIs' secret keys from .env file.
secret_keys = dotenv_values(".env")
HF_TOKEN= secret_keys["HF_TOKEN"]


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def create_bnb_config():
    """
    Configures model quantization method using bitsandbytes to speed up training and inference
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.bfloat16,
    )

    return bnb_config


def load_model(model_name, bnb_config):
    """
    Loads model and model tokenizer

    :param model_name: Hugging Face model name
    :param bnb_config: Bitsandbytes configuration
    """

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        device_map = "auto", # dispatch the model efficiently on the available resources
        # max_memory = {i: max_memory for i in range(n_gpus)},
        token= HF_TOKEN,
        use_cache=False,
    )
    model.config.pretraining_tp = 1

    # Load model tokenizer with the user authentication token
    tokenizer = AutoTokenizer.from_pretrained(model_name, token= HF_TOKEN)

    # Set padding token as EOS token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def create_peft_config(r, lora_alpha, lora_dropout, target_modules):
    """
    Creates Parameter-Efficient Fine-Tuning configuration for the model

    :param r: LoRA attention dimension
    :param lora_alpha: Alpha parameter for LoRA scaling
    :param lora_dropout: Dropout Probability for LoRA layers
    """
    config = LoraConfig(
        r = r,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        target_modules = target_modules,
        bias = "none",
        task_type = "CAUSAL_LM",
    )

    return config

    
def find_all_linear_names(model):
    """
    Find modules to apply LoRA to.

    :param model: PEFT model
    """

    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    print(f"LoRA module names: {list(lora_module_names)}")
    return list(lora_module_names)


def print_trainable_parameters(model, use_4bit = True):
    """
    Prints the number of trainable parameters in the model.

    :param model: PEFT model
    """

    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    if use_4bit:
        trainable_params /= 2

    print(
        f"All Parameters: {all_param:,d} || Trainable Parameters: {trainable_params:,d} || Trainable Parameters %: {100 * trainable_params / all_param}"
    )


def fine_tune(model, tokenizer, dataset, max_seq_length, lora_r, lora_alpha, lora_dropout, training_args, task_type):
    """
    Prepares and fine-tune the pre-trained model.

    :param model: Pre-trained Hugging Face model
    :param tokenizer: Model tokenizer
    :param dataset: Preprocessed training dataset
    """

    # Enable gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # Prepare the model for training
    model = prepare_model_for_kbit_training(model)

    # Get LoRA module names
    target_modules = find_all_linear_names(model)

    # Create PEFT configuration for these modules and wrap the model to PEFT
    peft_config = create_peft_config(lora_r, lora_alpha, lora_dropout, target_modules)
    model = get_peft_model(model, peft_config)

    # Print information about the percentage of trainable parameters
    # print_trainable_parameters(model)

    # Training parameters

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        # tokenizer=tokenizer,
        packing=True,
        args=training_args,
        dataset_text_field="text",
        # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
        )


    # Launch training and log metrics
    print("Training...")

    trainer_stats = trainer.train()
    train_loss = trainer_stats.training_loss
    print(f"Training loss:{train_loss}")

    print("-"* 50 , "After Training ", "-"* 50)
    print_gpu_utilization()

    # Save model
    peft_model_id = f"{training_args.output_dir}/assets"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    with open(f"{training_args.output_dir}/results.pkl", "wb") as handle:
        run_result = [
            training_args.num_train_epochs,
            lora_r,
            lora_dropout,
            train_loss,
        ]
        pickle.dump(run_result, handle)
    print("Experiment over")

    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-learning_rate",default=2e-4, type=float, help="learning rate")
    parser.add_argument("-seed",default=14, type=int, help="seed to replicate results")
    parser.add_argument("-model_name", default="mistralai/Mistral-7B-v0.1", help = "The name of target model")
    parser.add_argument("-task_type", default="veracity", help = "The target task", choices=['veracity', 'explanation', 'joint'])
    parser.add_argument("-lora_r",default=16, type=int, help="LoRA attention dimension")
    parser.add_argument("-lora_alpha",default=16, type=int, help="Alpha parameter for LoRA scaling")
    parser.add_argument("-lora_dropout",default=0.1, type=float, help="Dropout probability for LoRA layers")
    parser.add_argument("-per_device_train_batch_size",default=64, type=int, help="Batch size per GPU for training")
    parser.add_argument("-gradient_accumulation_steps",default=1, type=int, help="Number of update steps to accumulate the gradients for")
    parser.add_argument("-optimizer",default="paged_adamw_32bit", type=str, help="Optimizer to use")
    parser.add_argument("-epochs",default=30, type=int, help="Number of training epochs)")
    parser.add_argument("-max_seq_length",default=830, type=int, help="max sequence length for model and packing of the dataset")
    parser.add_argument("-weight_decay",default=0.0, type=float, help="The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer")

    args = parser.parse_args()

    print("-"* 100)
    print(f"Start training for {args.task_type} {args.model_name} epochs: {args.epochs} lora_r: {args.lora_r} lora_dropout: {args.lora_dropout}")

    if torch.cuda.is_available():
        print("-"*50, "Cuda is available!", "-"*50)
        print("-"*45, "device_count:",torch.cuda.device_count(), "-"*45)
    else:
        print("-"*50, "No CUDA!!!")

    print_gpu_utilization()

    print("Start Loading the Model")
    # Load model from Hugging Face Hub with model name and bitsandbytes configuration
    bnb_config = create_bnb_config()
    model, tokenizer = load_model(args.model_name, bnb_config)

    base_dir= os.path.dirname(os.getcwd())
    remove_columns= ['claim_id', 'claim', 'explanation', 'summarized_text', 'label']
    # Train set
    dataset= read_dataset(file_path=base_dir+"/data/pubhealth/summarization/train_chatgpt_summarized.csv", task_type= args.task_type)
    preprocessed_dataset = preprocess_dataset(tokenizer, args.max_seq_length, args.seed, dataset, remove_columns)

    # Output directory where the model predictions and checkpoints will be stored
    output_dir = f"results/{args.task_type}/{args.model_name}/{args.epochs}_rank-{args.lora_r}_dropout-{args.lora_dropout}"
    
    training_args= TrainingArguments(
        per_device_train_batch_size = args.per_device_train_batch_size,
        # gradient_accumulation_steps = args.gradient_accumulation_steps,
        num_train_epochs = args.epochs,
        learning_rate = args.learning_rate,
        bf16=True, # Enable fp16/bf16 training (set bf16 to True with an A100)
        tf32=True,
        logging_steps=100,
        output_dir = output_dir,
        logging_dir=f"{output_dir}/logs",
        optim = args.optimizer,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="none",
        save_strategy= "epoch",
        seed= args.seed,
        weight_decay= args.weight_decay 
        )

    print("-"* 50 , "After Loading Models and Reading Dataset ", "-"* 50)
    print_gpu_utilization()

    fine_tune(model, tokenizer, preprocessed_dataset, args.max_seq_length, args.lora_r
        , args.lora_alpha, args.lora_dropout, training_args, args.task_type)


if __name__ == '__main__':
	main()