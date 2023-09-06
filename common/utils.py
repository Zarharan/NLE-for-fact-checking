import requests
import openai
from openai.error import RateLimitError
import backoff
from data.pubhealth.models import *
import math
from torchmetrics.text.bert import BERTScore
from torchmetrics import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
import nltk
from datetime import timezone, datetime
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import spacy
import numpy as np
from common.nli_structure import *
import torch
# from allennlp_models import pretrained
import logging
from dotenv import dotenv_values
import random
from tqdm import tnrange, trange
import torch.nn.functional as F


# nltk.download('punkt')
PROMPT_TEMPLATES = {
    "PubHealth": {
        "veracity" : {
            "basic": ("Context: {}\nClaim: {}\nWhich of true, false, mixture, and unproven can be the label of the claim by considering the context? {}\n"),
            "natural": ("Context: {}\nClaim: {}\nTaking into consideration the context of the claim, label the claim as either true, false, mixture, or unproven. {}\n"),
            "bias_checking": ("{}Claim: {}\nWhich of true, false, mixture, and unproven can be the label of the claim? {}\n"),
        },
        "explanation": {
            "basic": ("Context: {}\nClaim: {}\nclaim is {}\nWhy? {}\n"),
            "natural": ("Context: {}\nClaim: {}\nclaim is {}\nExplain the veracity of the claim by considering just the related context. {}\n"),
            "bias_checking": ("{}Claim: {}\nclaim is {}\nExplain the veracity of the claim. {}\n")
        },
        "joint":{
            "basic": ("Context: {}\nClaim: {}\n Which of true, false, mixture, and unproven can be the label of the claim by considering the context? {}\nWhy? {}\n"),
            "natural": ("Context: {}\nClaim: {}\n Predict the veracity of the claim and explain your reasoning by considering just the related context. Assign one of true, false, mixture, or unproven as the veracity label of the claim.\n {} \n{}"),
            "bias_checking": ("{}Claim: {}\n Predict the veracity of the claim and explain your reasoning. Assign one of true, false, mixture, or unproven as the veracity label of the claim.\n {} \n{}"),
        }

    }
}

CHAT_COMPLETION_SYSTEM_ROLE= {
    "veracity": "You are a helpful assistant that predicts the veracity of a claim by considering the context. Instructions: - Predict the veracity of claims by considering just the related context. - Assign one of True, False, Mixture, or Unproven as the veracity label of the claim.",
    "explanation": "You are a helpful assistant that explains the veracity of a claim by considering the context. Instructions: - Explain the veracity of a claim by considering just the related context.",
    "joint":"You are a helpful assistant that predicts the veracity of a claim and explains the reason for your prediction by considering the context. Instructions: - Predict the veracity of a claim and explain your reasoning by considering just the related context. - Assign one of True, False, Mixture, or Unproven as the veracity label of the claim.",
    "veracity/bias_checking": "You are a helpful assistant that predicts the veracity of a claim. Instructions: - Predict the veracity of claims. - Assign one of True, False, Mixture, or Unproven as the veracity label of the claim.",
    "explanation/bias_checking": "You are a helpful assistant that explains the veracity of a claim. Instructions: - Explain the veracity of a claim.",
    "joint/bias_checking":"You are a helpful assistant that predicts the veracity of a claim and explains the reason for your prediction. Instructions: - Predict the veracity of a claim and explain your reasoning. - Assign one of True, False, Mixture, or Unproven as the veracity label of the claim."
}

CHATGPT_EXTRA_DESC= {
    "zero": "Can you please explain the veracity of the following claim by considering the context?\n",
    "few": "The following are some examples of explanations for the veracity of a claim. The context for each claim is provided. Can you please explain the veracity of the last claim by considering its context?\n"
}

NLI_LABEL_ID= {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}


# take Hugging face and Open AI APIs' secret keys from .env file.
secret_keys = dotenv_values(".env")
HF_TOKEN= secret_keys["HF_TOKEN"]
OAI_API_KEY= secret_keys["OAI_API_KEY"]

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


def get_utc_time():
    '''
    This function returns UTC timestamp
    '''    
    dt = datetime.now(timezone.utc)
    
    utc_time = dt.replace(tzinfo=timezone.utc)
    return utc_time.timestamp()


def add_chatgpt_prompt(target_instances, prompt_type):
    '''
    This function adds another coloumn to the result file for ChatGPT prompt
    
    :param target_instances: The input instances to create prompt for them
    :type target_instances: str
    :param prompt_type: The type of the prompt which is zero or few
    :type prompt_type: str

    :returns: The input instance list with ChatGPT prompts added to each instance
    :rtype: list
    '''

    for target_instance in target_instances:
        target_instance['chatgpt_prompt'] = CHATGPT_EXTRA_DESC[prompt_type] + target_instance['prompt']
            
    return target_instances


def sent_tokenize(input_text):    
    '''
    This function returns the list of sentences in the input text

    :param input_text: The input text
    :type input_text: str

    :returns: The list of sentences in the input text
    :rtype: list
    '''

    nlp= spacy.load("en_core_web_sm")
    doc= nlp(input_text)

    lst_sent= [x.text for x in doc.sents]
    return lst_sent


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


class Summarization():
    '''
    The Summarization object is responsible for implementing different methods to summarize a text (e.g. main text of the news).

    :param max_tokens: The maximum number of tokens for generated summary
    :type max_tokens: str
    :param temperature: To set the randomness of generated text (between 0 and 1, with 0 being the most predictable and 1 being the most random)
    :type temperature: float
    :param model_name: The target model to generate summary
    :type model_name: string
    :param model_path: The path of weights of the target model to generate summary (except GPT-3 model)
    :type model_path: string

    :ivar _summarizer_tokenizer: The loaded tokenizer to prevent loading several times
    :vartype _summarizer_tokenizer: object
    :ivar _summarizer: The loaded summarizer model to prevent loading several times
    :vartype _summarizer: object
    :ivar text_summary: The object to do CRUD queries on the summaries table
    :vartype text_summary: object
    :ivar _device: The target device (CPU or GPU) on which the model will be loaded.
    :vartype _device: object    
    '''
    def __init__(self, max_tokens= 300, temperature= 0.5, model_name= "bart"
        , model_path="data/models/bart"):

        self.max_tokens= max_tokens
        self.temperature= temperature
        self.model_name= model_name
        self.model_path= model_path
        self._summarizer= None
        self._summarizer_tokenizer= None
        self.text_summary = TextSummary()

        self._device= torch.device('cpu')

    
    def get_summary(self, text_for_summary, claim_id):
        ''' This function gets a text and returns the summary. 
        If for the input claim, a summary exists in the database, it is returned.
        If not, generates a summary and saves and returns

        :param text_for_summary: The input text 
        :type text_for_summary: str
        :param claim_id: The target claim Id 
        :type claim_id: int

        :returns: The generated summary
        :rtype: str
        '''

        # Read and return summary from database if for the target claim, the summary was already generated by the target model
        select_result= self.text_summary.select_summary(claim_id, self.model_name)
        if select_result:
            return select_result.summary

        # 1 token ~= ¾ words. 100 tokens ~= 75 words (Regarding OpenAI documentation)
        # check the max tokens of the text and truncate if exceed
        main_text_words= text_for_summary.split()
        tolerance_no= 500
        exceed_no= 4096 - (len(main_text_words) * (4/3) + self.max_tokens + tolerance_no)
        if exceed_no<0:
            text_for_summary= " ".join(main_text_words[:math.floor(exceed_no)])

        summary = ""
        if self.model_name== "bart":
            summary= self.__bart_large_cnn(text_for_summary, int(exceed_no))
        elif self.model_name== "lsg_bart":
            summary= self.__lsg_bart_large(text_for_summary)
        elif self.model_name == "gpt3":
            summary= self.__gpt3(text_for_summary)

        # Save summary into the related table for later
        summary_data= SummaryModel(claim_id= claim_id, main_text= text_for_summary
            , summary= summary, model_name= self.model_name)
        self.text_summary.insert(summary_data)        
        return summary


    @backoff.on_exception(backoff.expo, RateLimitError)
    def __gpt3(self, text_for_summary):
        ''' This function gets a text and summarizes it by generating at most max_tokens by using GPT-3.

        :param text_for_summary: The input text 
        :type text_for_summary: str

        :returns: The generated summary
        :rtype: str
        '''

        openai.api_key= OAI_API_KEY
        text_for_summary+= "\nTL;DR:\n"
        response = openai.Completion.create(engine="text-davinci-003", prompt=text_for_summary
            , temperature= self.temperature,max_tokens=self.max_tokens, top_p=1, frequency_penalty=0, presence_penalty=0)

        return response.choices[0].text


    def __bart_large_cnn(self, text_for_summary, max_length):
        ''' This function gets a text and summarizes it by generating at most max_tokens by using bart-large-cnn-samsum from HiggingFace.

        :param text_for_summary: The input text 
        :type text_for_summary: str
        :param max_length: The max token counts for the input text 
        :type max_length: int

        :returns: The generated summary
        :rtype: str
        '''      
        
        if self._summarizer is None or self._summarizer_tokenizer is None: 
            if torch.cuda.is_available():
                self._device = torch.device('cuda')
                print("The cuda is available.\n")
            else:
                print("The cuda is not available.\n")

            # only load for first use
            self._summarizer_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._summarizer = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(self._device)
            print("Bart model was loaded successfully.")

        inputs = self._summarizer_tokenizer([text_for_summary], max_length=max_length
        , truncation=True, return_tensors="pt")
        transfered_input_ids = {k: v.to(self._device) for k, v in inputs.items()}       
        summary_ids = self._summarizer.generate(transfered_input_ids["input_ids"], num_beams=2, min_length=0, max_length=self.max_tokens)
        return self._summarizer_tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


    def __lsg_bart_large(self, text_for_summary):
        ''' This function gets a text and summarizes it by generating at most max_tokens by using ccdv/lsg-bart-base-4096-multinews from HiggingFace.

        :param text_for_summary: The input text 
        :type text_for_summary: str

        :returns: The generated summary
        :rtype: str
        '''      

        if self._summarizer is None or self._summarizer_tokenizer is None: 
            if torch.cuda.is_available():
                self._device = torch.device('cuda')
                print("The cuda is available.\n")
            else:
                print("The cuda is not available.\n")

            # only load for first use
            self._summarizer_tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self._summarizer = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, trust_remote_code=True).to(self._device)
            print("LSG Bart model was loaded successfully.")

        inputs = self._summarizer_tokenizer([text_for_summary], return_tensors="pt")        
        transfered_input_ids = {k: v.to(self._device) for k, v in inputs.items()}       
        summary_ids = self._summarizer.generate(transfered_input_ids["input_ids"], num_beams=2, max_length=self.max_tokens)
        return self._summarizer_tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


class NLEMetrics():
    '''
    The NLEMetrics object is responsible for obtaining different metrics to evaluate a generated explanation.

    :param pred_list: The list of generated explanation
    :type pred_list: list
    :param target_list: The list of ground truth explanation
    :type target_list: list
    :param bertscore_model: The model to calculate the BERTScore
    :type bertscore_model: string
    :param claim_list: The list of claims
    :type claim_list: list
    :param claim_gold_label_list: The list of ground truth label of claims
    :type claim_gold_label_list: list
    :param nli_model: The NLI model to calculate coherence metrics (an object of a class inherited from NLIStructure)
    :type nli_model: object

    :ivar rouge: The object to calculate the rouge score
    :vartype rouge: object
    :ivar bertscore: The object to calculate the BERTScore
    :vartype bertscore: object
    :ivar bleu: The object to calculate the bleu score
    :vartype bleu: object        
    '''
    def __init__(self, pred_list = None, target_list= None
        , bertscore_model= "microsoft/deberta-xlarge-mnli", claim_list= [], claim_gold_label_list=[], nli_model= None):
        
        self.pred_list= pred_list
        self.target_list= target_list
        self.bertscore_model= bertscore_model
        self.claim_list= claim_list
        self.claim_gold_label_list= claim_gold_label_list
        self.nli_model= nli_model

        self.rouge= None
        self.bertscore= None
        self.bleu= None


    def rouge_score(self):
        ''' This function calculate the rouge score for pred_list regarding target_list.

        :returns: The average rouge score for the list
        :rtype: float
        '''

        if self.rouge is None:
            self.rouge = ROUGEScore()
            
        log("Start calculating ROUGE score ...")
        rouge_result= self.rouge(self.pred_list, self.target_list)
        log(rouge_result)
        return rouge_result

    
    def bert_score(self):
        ''' This function calculate the BERTScore for pred_list regarding target_list.

        :returns: The average BERTScore for the list
        :rtype: float
        '''

        if self.bertscore is None:
            self.bertscore = BERTScore(model_type= self.bertscore_model)

        log("Start calculating BERT score ...")
        score = self.bertscore(self.pred_list, self.target_list)
        rounded_score = {k: [round(v, 4) for v in vv] for k, vv in score.items()}
        log(f"BERT Score: {rounded_score}")
        return rounded_score


    def bleu_score(self):
        ''' This function calculate the bleu score for pred_list regarding target_list.

        :returns: The average bleu score for the list
        :rtype: float
        '''

        if self.bleu is None:
            self.bleu = BLEUScore()        
        
        bleu_avg= 0
        log("Start calculating BLEU score ...")
        target_count= len(self.target_list)
        # Calculate the average bleu score for all instances in the list
        for index, (pred, target) in enumerate(zip(self.pred_list, self.target_list)):
            if (index+1) % 100 == 0:
                log(f"-------- {index+1}/{target_count} --------")            
            bleu_avg+= self.bleu([pred], [[target]]).item()

        rounded_score = round(bleu_avg / target_count, 4)
        log(f"BLEU Score: {rounded_score}")
        return rounded_score


    def __check_coherence_inputs(func):
        '''
        This is a decorator to check the required inputs of coherence functions.
        '''
        def wrapper(self, *args, **kwargs):

            assert len(self.claim_list) > 0, "Please set the related claim list to the predicted/generated list"
            assert self.nli_model is not None, "Please set the NLI object for inference"

            func_result = func(self, *args, **kwargs)
            return func_result

        return wrapper


    @__check_coherence_inputs
    def SGC(self):
        ''' This function calculates the strong global coherence metric by following the definition of the "Explainable Automated Fact-Checking for Public Health Claims" paper.
        The definition: Every sentence in the generated explanation text must entail the claim.

        :returns: The percentage of instances that satisfy this metric.
        :rtype: float
        '''        
        failed_no= 0
        log("Start calculating SGC score ...")
        target_count= len(self.claim_list)
        for index, (claim, pred) in enumerate(zip(self.claim_list, self.pred_list)):
            if (index+1) % 100 == 0:
                log(f"-------- {index+1}/{target_count} --------")
                
            pred_sents_list= sent_tokenize(pred)
            for sent in pred_sents_list:
                if self.nli_model.predict_nli(sent, claim) != NLI_LABEL_ID["entailment"]:
                    failed_no+= 1
                    break
        
        rounded_score = round(1-(failed_no/target_count), 4)
        log(f"SGC Score: {rounded_score}")
        return rounded_score


    @__check_coherence_inputs
    def WGC(self):
        ''' This function calculates the weak global coherence metric by following the definition of the "Explainable Automated Fact-Checking for Public Health Claims" paper.
        The definition: No sentence in the generated explanation text should contradict the claim.

        :returns: The percentage of instances that satisfy this metric.
        :rtype: float
        '''
        assert len(self.claim_gold_label_list) > 0, "Please set the claim's ground truth label list"

        failed_no= 0
        log("Start calculating WGC score ...")
        target_count= len(self.claim_list)
        for index, (claim, pred, claim_label) in enumerate(zip(self.claim_list, self.pred_list, self.claim_gold_label_list)):
            if (index+1) % 100 == 0:
                log(f"-------- {index+1}/{target_count} --------")
                
            pred_sents_list= sent_tokenize(pred)
            for sent in pred_sents_list:
                if self.nli_model.predict_nli(sent, claim) == NLI_LABEL_ID["contradiction"] and claim_label.strip().lower() != "false":
                    failed_no+= 1
                    break
        
        rounded_score = round(1-(failed_no/target_count), 4)
        log(f"WGC Score: {rounded_score}")
        return rounded_score


    @__check_coherence_inputs
    def LC(self):
        ''' This function calculates the local coherence metric by following the definition of the "Explainable Automated Fact-Checking for Public Health Claims" paper.
        The definition: Any two sentences in the generated explanation text must not contradict each other.

        :returns: The percentage of instances that satisfy this metric.
        :rtype: float
        '''
        failed_no= 0
        log("Start calculating LC score ...")
        target_count= len(self.pred_list)

        def _locally_coherent(pred):
            pred_sents_list= sent_tokenize(pred)
            for sent1 in pred_sents_list:
                for sent2 in pred_sents_list:
                    if sent1 != sent2 and self.nli_model.predict_nli(sent1, sent2) == NLI_LABEL_ID["contradiction"]:
                        return False

            return True

        for index, pred in enumerate(self.pred_list):
            if (index + 1) % 100 == 0:
                log(f"-------- {index + 1}/{target_count} --------")
            failed_no += int(not _locally_coherent(pred, ))

        rounded_score = round(1-(failed_no/target_count), 4)
        log(f"LC Score: {rounded_score}")
        return rounded_score


    def get_all_metrics(self):
        ''' This function calculate all scores to evaluate the pred_list regarding the target_list.

        :returns: The average score for all metrics
        :rtype: dict
        '''

        return {"rouge": self.rouge_score(), "SGC": self.SGC(), "WGC": self.WGC(), "LC": self.LC(), "bleu": self.bleu_score()}
                

class NLI(NLIStructure):
    '''
    The NLI object is responsible for obtaining NLI labelof an instance.

    :param model_name: The model name to predict NLI label
    :type model_name: str
    :param model_path: The path of selected model to load
    :type model_path: str

    :ivar _nli_tokenizer: The loaded tokenizer to prevent loading several times
    :vartype _nli_tokenizer: object
    :ivar _nli_model: The loaded NLI model to prevent loading several times
    :vartype _nli_model: object
    :ivar _model_func_mapping: The mapping between selected model name and related implemented function
    :vartype _model_func_mapping: dict
    :ivar _device: The target device (CPU or GPU) on which the model will be loaded.
    :vartype _device: object
    '''
    def __init__(self, model_name, model_path):
            
        self._model_func_mapping= {"roberta_large_snli": self.__roberta_large_snli, "allennlp_nli_models": self.__allennlp_nli_models}
        assert model_name in self._model_func_mapping.keys(), f"Please select one of {self._model_func_mapping.keys()} as target NLI model or add a new model name and related implementation."
        self.model_name= model_name
        self.model_path= model_path

        self._nli_tokenizer= None
        self._nli_model= None
        self._device= torch.device('cpu')


    def predict_nli(self, premise, hypothesis):
        ''' This function returns the NLI label ID with "entailment": 0, "neutral": 1, and "contradiction": 2.

        :param premise: The premise
        :type premise: str
        :param hypothesis: The hypothesis
        :type hypothesis: str

        :returns: The NLI label ("entailment": 0, "neutral": 1, and "contradiction": 2)
        :rtype: int
        '''
        
        return self._model_func_mapping[self.model_name](premise, hypothesis)


    def __roberta_large_snli(self, premise, hypothesis):
        ''' This function returns the NLI label ID with "entailment": 0, "neutral": 1, and "contradiction": 2.
        It uses roberta_large_snli (ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli) transformer to predict the label.

        :param premise: The premise
        :type premise: str
        :param hypothesis: The hypothesis
        :type hypothesis: str

        :returns: The NLI label ("entailment": 0, "neutral": 1, and "contradiction": 2)
        :rtype: int
        '''

        if self._nli_model is None or self._nli_tokenizer is None:
            if torch.cuda.is_available():
                self._device = torch.device('cuda')
                print("The cuda is available.\n")
            else:
                print("The cuda is not available.\n")

            self._nli_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._nli_model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(self._device)

        input_ids= self._nli_tokenizer.encode_plus(premise, hypothesis, max_length=200
        , truncation=True, return_tensors="pt")
        transfered_input_ids = {k: v.to(self._device) for k, v in input_ids.items()}
        outputs = self._nli_model(**transfered_input_ids)

        return np.argmax(torch.softmax(outputs[0], dim=1)[0].tolist())


    def __allennlp_nli_models(self, premise, hypothesis):
        ''' This function returns the NLI label ID with "entailment": 0, "neutral": 1, and "contradiction": 2.
        It uses the NLI model that is set in self.model_path (self.model_path value options: pair-classification-decomposable-attention-elmo, pair-classification-roberta-mnli, or pair-classification-roberta-snli) to predict the label.

        :param premise: The premise
        :type premise: str
        :param hypothesis: The hypothesis
        :type hypothesis: str

        :returns: The NLI label ("entailment": 0, "neutral": 1, and "contradiction": 2)
        :rtype: int
        '''

        if self._nli_model is None:
            self._nli_model = pretrained.load_predictor(self.model_path)
        
        result= self._nli_model.predict_json({ "premise": premise, "hypothesis": hypothesis})
        return NLI_LABEL_ID[result["label"]]