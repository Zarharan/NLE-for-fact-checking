from datetime import timezone, datetime
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import numpy as np
import torch
import logging
import random
from tqdm import tnrange, trange
import torch.nn.functional as F


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
# End of logging


def log(*args):
    '''
    This function print and log all input arguments into a file.
    '''      
    for arg in args:
        print(arg)
        logger.info(arg)


def tokenizer_with_special_tokens(tokenizer_name):
    ''' Returns the tokenizer after adding separator and padding tokens

    :param tokenizer_name: The name of target tokenizer like "EleutherAI/gpt-j-6b", "gpt2-xl", etc
    :type tokenizer_name: str

    :returns: The target tokenizer
    :rtype: object
    '''
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    special_tokens = {'pad_token': tokenizer.eos_token,'sep_token':'<|sep|>'}
    num_add_toks = tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def set_seed(seed, n_gpu= 0):
    '''
    This function sets the seed

    :param seed: The seed number
    :type seed: int
    :param n_gpu: The number og gpus
    :type n_gpu: int    

    :returns None
    '''    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_seq(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    is_xlnet=False, is_xlm_mlm=False, xlm_mask_token=None, xlm_lang=None, device='cpu'):
    """ Generates a sequence of tokens 
        Args:
            model: The selected GPT model
            length: length of generated sequence.
            context: tokenized text using gpt/gpt2 tokenizer
            num_samples: number of returned samples
            device: torch.device object.
            temperature >0: used to control the randomness of predictions by scaling the logits before applying softmax.
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            repetition_penalty: The penalty for repetition.

            from: https://github.com/huggingface/transformers/blob/18a879f47576822aa1a5c49aecb27d89bfa5fa69/examples/run_generation.py
    """                    
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):

            inputs = {'input_ids': generated}
            if is_xlnet: 
                # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                input_ids = torch.cat((generated, torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
                perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
                perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                target_mapping[0, 0, -1] = 1.0  # predict last token
                inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

            if is_xlm_mlm and xlm_mask_token:
                # XLM MLM models are direct models (predict same token, not next token)
                # => need one additional dummy token in the input (will be masked and guessed)
                input_ids = torch.cat((generated, torch.full((1, 1), xlm_mask_token, dtype=torch.long, device=device)), dim=1)
                inputs = {'input_ids': input_ids}

            if xlm_lang is not None:
                inputs["langs"] = torch.tensor([xlm_lang] * inputs["input_ids"].shape[1], device=device).view(1, -1)

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty
                
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated


def generate_sample(data, tokenizer, model, num=1, eval_step=False, length=60, temperature=1, top_k=10, top_p=0.5, device=torch.device('cuda')):
    """ Generate explanation for "num" number of instances.
        Args:
            data = validation set or test set
            tokenizer = The selected tokenizer.
            model = The selected GPT model
            num = number of instances for which explanation has to be generated
            eval_step = can be True/False, checks generating during evaluation or not
            length: length of generated sequence.
            temperature >0: used to control the randomness of predictions by scaling the logits before applying softmax.
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            device: torch.device object.
    """
    for i in range(num):
        sample = data[i]
        # the length of the context section (claim and text) of the input text
        idx = sample['target_len']
        context = sample['input_ids'][0][:idx].tolist()
        label = sample['input_ids'][0][idx+1:].tolist()
        generated_text = sample_seq(model= model, length= length, context= context, temperature= temperature, top_k= top_k, top_p= top_p, device= device)
        generated_text = generated_text[:, idx:].tolist()

        text= []
        for result in generated_text:
            text.append(tokenizer.decode(result,skip_special_tokens=True))

        if eval_step==False:
            log('main text: ', '\n\n')
            log(tokenizer.decode(context), '\n\n')
            log("generated results", '\n\n')
            log(text, '\n\n')
            log('actual explanation', '\n\n')
            log(tokenizer.decode(label), '\n\n')
        else:
            log('main text: ', '\n\n')
            log(tokenizer.decode(context), '\n\n')
            log("generated results", '\n\n')
            log(text, '\n\n')    

