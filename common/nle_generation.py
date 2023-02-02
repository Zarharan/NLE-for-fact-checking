from common.utils import *
import json


class NLEGeneration():

  def __init(self):
    pass


  def __prompt(self, target_instances, pattern, type="zero"):
    ''' This function creates appropriate prompt for the input instances.

    :param target_instances: The input instances to create prompt for them
    :type target_instances: str
    :param pattern: The pattern to apply on the input text to create prompt
    :type pattern: str
    :param type: By using zero, the prompt does not include the gold explanation
    :type type: str

    :returns: An input instance list with prompts added to each instance
    :rtype: list
    '''

    for target_instance in target_instances:
      target_instance['prompt'] =pattern.format(target_instance['main_text'], target_instance['claim']
        , target_instance['label'], "" if type=="zero" else target_instance['explanation'])
            
    return target_instances


  def __gpt3_query(self, prompt, max_tokens):
    ''' This function send a query to GPT-3.

    :param prompt: The target prompt 
    :type prompt: str
    :param max_tokens: The maximum number of tokens for generated text
    :type max_tokens: str

    :returns: The generated text
    :rtype: str
    '''

    openai.api_key= OAI_API_KEY
    response = openai.Completion.create(engine="text-davinci-002", prompt= prompt, temperature=0.2
      , max_tokens=max_tokens, top_p=1, frequency_penalty=0, presence_penalty=0)
    
    return response.choices[0].text


  def __hf_models_query(self, prompt, max_tokens, model_name):
    ''' This function send a query to HuggingFace models.

    :param prompt: The target prompt 
    :type prompt: str
    :param max_tokens: The maximum number of tokens for generated text
    :type max_tokens: str
    :param model_name: The name of HuggingFace model
    :type model_name: str

    :returns: The generated text
    :rtype: str
    '''

    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    parameters = {
        'max_new_tokens':max_tokens,  # number of generated tokens
        'temperature': 0.2,   # controlling the randomness of generations
        'end_sequence': "###" # stopping sequence for generation
    }
    options={'use_cache': True}
    body = {"inputs":prompt,'parameters':parameters,'options':options}
    response = requests.post(API_URL, headers=headers, json= body)

    return response.json()[0]['generated_text']


  def gpt3_zero_shot(self, target_instances, pattern, max_tokens= 200):
    ''' This function explain veracity of instances by using GPT-3 zero-shot.

    :param target_instances: The input text 
    :type target_instances: str
    :param pattern: The target pattern to create the prompt 
    :type pattern: str    
    :param max_tokens: The maximum number of tokens for generated text
    :type max_tokens: str

    :returns: An input instance list with generated explanation added to each instance
    :rtype: list
    '''
    
    # create appropriate prompt
    self.__prompt(target_instances, pattern, type="zero")

    for target_instance in target_instances:
      target_instance['result']= self.__gpt3_query(target_instance['prompt'], max_tokens)
    
    return target_instances


  def gpt_j_zero_shot(self, target_instances, pattern, max_tokens= 200):
    ''' This function explain veracity of instances by using gpt-j-6B zero-shot.

    :param target_instances: The input text 
    :type target_instances: str
    :param pattern: The target pattern to create the prompt 
    :type pattern: str    
    :param max_tokens: The maximum number of tokens for generated text
    :type max_tokens: str

    :returns: An input instance list with generated explanation added to each instance
    :rtype: list
    '''
    
    # create appropriate prompt
    self.__prompt(target_instances, pattern, type="zero")

    for target_instance in target_instances:       
      target_instance['result']= self.__hf_models_query(target_instance['prompt'], max_tokens
        , "EleutherAI/gpt-j-6B")        
      target_instance['result']= response.json()[0]['generated_text']
            
    return target_instances    


