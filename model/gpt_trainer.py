from datetime import datetime
import os
import time
import numpy as np
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tnrange, tqdm
from model.generation_utils import *


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
    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dl, desc="Training")
        for step, batch in enumerate(epoch_iterator):
            inputs, labels = torch.tensor(batch['input_ids']), torch.tensor(batch['input_ids'])
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            logits = model(inputs)[0]
            idx = batch['target_len'].item() # the length of the context section (claim and text) of the input text
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
                                
            if (step + 1) % (100*args.gradient_accumulation_steps) == 0:
                results = evaluate(args, model, val_dl, ignore_index, global_step)
                for key, value in results.items():
                    writer.add_scalar('eval_{}'.format(key), value, global_step)

        log(f'After {(epoch + 1)} epoch(s): ', '\n\n')
        generate_sample(val_dataset, tokenizer, model, num=1, eval_step=False,device=args.device,
            top_k= args.top_k, top_p= args.top_p, length= args.generation_length, temperature= args.temperature)
                         

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
            idx = batch['target_len'].item() # the length of the context section (claim and text) of the input text
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