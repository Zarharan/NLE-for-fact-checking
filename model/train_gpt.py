import argparse
from datetime import datetime
import os
import time
import numpy as np
from transformers import AutoModelForCausalLM,AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tnrange, tqdm
from model.common.dataset import PubHealthDataset
from common.utils import *
from pathlib import Path


# logging
file_handler = logging.FileHandler('train_gpt.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# End of logging


def train(args, model, tokenizer, train_dl, val_dl, val_dataset, ignore_index):
    '''
    This function Trains the selected GPT model and logs necessary details.

    :param args: The object that contains all the necessary information passed by user while training
    :type args: object
    :param model: The loaded model
    :type model: object
    :param tokenizer: The target tokenizer
    :type tokenizer: object
    :param train_dl: The training data loader for training model
    :type train_dl: object
    :param val_dl: The validation data loader for evaluating model
    :type val_dl: object
    :param val_dataset: The validation dataset object
    :type val_dataset: object
    :param ignore_index: The token not considered in loss calculation
    :type ignore_index: object

    :returns None
    '''    
    writer = SummaryWriter('./logs')
    
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index) #ignores padding token for loss calculation
    optimizer = AdamW(model.parameters(),lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer,100,80000)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = tnrange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args.seed, args.n_gpu)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dl, desc="Training")
        for step, batch in enumerate(epoch_iterator):
            inputs, labels = torch.tensor(batch['input_ids']), torch.tensor(batch['input_ids'])
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            logits = model(inputs)[0]
            idx = batch['target_len'].item() # the length of the target section (label and explanation) of the input text
            # only consider loss on label and explanation section, just like seq2seq models
            shift_logits = logits[..., idx:-1, :].contiguous()
            shift_labels = labels[..., idx+1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss/args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                writer.add_scalar('loss', (tr_loss - logging_loss)/args.gradient_accumulation_steps, global_step)
                logging_loss = tr_loss
                log("loss:", loss.item(), '\n\n')
                if (step + 1)/args.gradient_accumulation_steps == 1.0:
                	log('After 1st update: ', '\n\n')
                	generate_sample(val_dataset, tokenizer, model, num=2, eval_step=False,device=args.device)
                                
            if (step + 1) % (10*args.gradient_accumulation_steps) == 0:
                results = evaluate(args, model, val_dl, ignore_index, global_step)
                for key, value in results.items():
                    writer.add_scalar('eval_{}'.format(key), value, global_step)
                log('After', global_step+1,'updates: ', '\n\n')
                generate_sample(val_dataset, tokenizer, model, num=2, eval_step=True,device=args.device)
                         

def evaluate(args, model, val_dl, ignore_index, global_step=None):
    '''
    This function evaluates the model by calculating perplexity score on validation set and logs necessary details.

    :param args: The object that contains all the necessary information passed by user while training
    :type args: object
    :param model: The loaded model
    :type model: object
    :param val_dl: The validation data loader for evaluating model
    :type val_dl: object    
    :param global_step: no. of times gradients have backpropagated
    :type global_step: int    
    :param ignore_index: The token not considered in loss calculation
    :type ignore_index: object

    :returns Returns perplexity score on validation dataset.
    :type dict
    '''    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    eval_output_dir = args.output_dir

    results = {}
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index) #ignores padding token for loss calculation

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(val_dl, desc="Evaluating"):
        inputs, labels = torch.tensor(batch['input_ids']).to(args.device), torch.tensor(batch['input_ids']).to(args.device)
        
        with torch.no_grad():
            logits = model(inputs)[0]
            idx = batch['target_len'].item() # the length of the target section (label and explanation) of the input text
            # only consider loss on label and explanation section, just like seq2seq models
            shift_logits = logits[..., idx:-1, :].contiguous()
            shift_labels = labels[..., idx+1:].contiguous()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }
    log("perplexity:", perplexity.item())

    if global_step:
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as f:
            for key in sorted(result.keys()):
                f.write('\n\n')
                f.write("time = %s, %s = %s, step = %s\n" % (datetime.now().strftime("%d/%m/%Y %H:%M:%S"), key, str(result[key]), str(global_step)))
    return result           


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr",default=5e-5, type=float, help="learning rate")
    parser.add_argument("-seed",default=313, type=int, help="seed to replicate results")
    parser.add_argument("-n_gpu",default=1, type=int, help="no of gpu available")
    parser.add_argument("-gradient_accumulation_steps",default=32, type=int, help="gradient_accumulation_steps")    
    parser.add_argument("-num_workers",default=4, type=int, help="num of cpus available")
    parser.add_argument("-device",default=torch.device('cpu'), help="torch.device object")
    parser.add_argument("-num_train_epochs",default=1, type=int, help="no of epochs of training")
    parser.add_argument("-output_dir",default='output', type=str, help="path to save evaluation results")
    parser.add_argument("-model_dir",default='weights', type=str, help="path to save trained model")
    parser.add_argument("-fp16",default=True, type=bool, help="whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("-fp16_opt_level",default='O0', type=str, help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
    parser.add_argument("-max_grad_norm",default=1.0, type=float, help="max gradient norm.")
    parser.add_argument("-train_path", help = "The tsv file path of the train set"
    , default='data/pubhealth/train.tsv', type= str)
    parser.add_argument("-val_path", default='data/pubhealth/dev.tsv', help = "The tsv file path of the validation set")
    parser.add_argument("-test_path", default='data/pubhealth/test.tsv', help = "The tsv file path of the test set")
    parser.add_argument("-truncation",default=True, type=bool, help="set truncation for the input text of the model")
    parser.add_argument("-main_text_max_length",default=1000, type=int, help="Set the max length for the main text section")
    parser.add_argument("-explanation_max_length",default=1000, type=int, help="Set the max length for the explanation section")
    parser.add_argument("-batch_size",default=1, type=int, help="batch_size")
    parser.add_argument("-tokenizer", default='EleutherAI/gpt-j-6b', help = "The name of target tokenizer")
    parser.add_argument("-model_name", default="EleutherAI/gpt-j-6b", help = "The name of target model")
    
    args = parser.parse_args()

    # Get the tokenizer with special tokens
    tokenizer = tokenizer_with_special_tokens(args.tokenizer)

    # Read the trining set and load into dataloader
    train_dataset = PubHealthDataset(data_path= args.train_path, tokenizer=tokenizer
        , truncation= args.truncation, main_text_max_length= args.main_text_max_length
        , explanation_max_length= args.explanation_max_length, pre_processor= None)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    log("The training Dataloader length= " + str(len(train_dataloader.dataset)))

    # Read the validation set and load into dataloader
    val_dataset = PubHealthDataset(data_path= args.val_path, tokenizer=tokenizer
        , truncation= args.truncation, main_text_max_length= args.main_text_max_length
        , explanation_max_length= args.explanation_max_length, pre_processor= None)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    log("The validation Dataloader length= " + str(len(val_dataloader.dataset)))
    
    # Load the model into the device
    ignore_idx = tokenizer.pad_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    start = time.time()

    # Train the model
    train(args, model, tokenizer, train_dataloader, val_dataloader, val_dataset, ignore_idx)
    log('total time: ', (time.time()-start)/60, " minutes", '\n\n')

    log('Saving trained model...')
    
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    model_file = os.path.join(args.model_dir, 'model_{}_data{}_trained_after_{}_epochs_only_sum_loss_ignr_pad.bin'.format(args.fp16_opt_level,3000,args.num_train_epochs))
    config_file = os.path.join(args.model_dir, 'config_{}_data{}_trained_after_{}_epochs_only_sum_loss_ignr_pad.json'.format(args.fp16_opt_level,3000,args.num_train_epochs))
    torch.save(model.state_dict(), model_file)
    model.config.to_json_file(config_file)


if __name__ == '__main__':
	main()
