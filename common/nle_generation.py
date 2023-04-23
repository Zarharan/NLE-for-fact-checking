from common.utils import *
import json
from openai.error import RateLimitError
import backoff


class NLEGeneration():
  '''
  By using NLEGeneration object we can generate explanation.
  
  :param prompt_template: The target template to create the prompt
  :type prompt_template: str
  :param temperature: To set the randomness of generated text (between 0 and 1, with 0 being the most predictable and 1 being the most random)
  :type temperature: float
  :param max_tokens: The maximum number of tokens for generated text
  :type max_tokens: int
  :param plm: The target pre-trained language model for few/zero shot learning
  :type plm: str  

  :ivar plms: List of possible pre-trained language models to select
  :vartype plms: list
  '''

  def __init__(self, prompt_template, plm="gpt3", temperature= 0.2, max_tokens= 200, plm_engine= ''):
    self.temperature= temperature
    self.max_tokens= max_tokens
    self.prompt_template= prompt_template
    self.selected_plm= plm
    self.plm_engine= plm_engine

    
    self.plms= {"gpt3": {"api_func": self.__openai_query, "engine": "text-davinci-003"}
      , "chat_gpt": {"api_func": self.__openai_chat_query, "engine": "gpt-3.5-turbo"}
      , "gptj": {"api_func": self.__hf_models_query, "model_name": "EleutherAI/gpt-j-6B"}}


  def __prompt(self, target_instances, type="zero"):
    ''' This function creates appropriate prompt for the input instances.

    :param target_instances: The input instances to create prompt for them
    :type target_instances: str
    :param type: By using zero, the prompt does not include the gold explanation
    :type type: str

    :returns: The input instance list with prompts added to each instance
    :rtype: list
    '''

    for target_instance in target_instances:
      target_instance['prompt'] = self.prompt_template.format(target_instance['summarized_main_text'], target_instance['claim']
        , target_instance['label'], "" if type=="zero" else target_instance['explanation'])

      if self.selected_plm == "chat_gpt" and type=="zero":
        target_instance['prompt']= [
                        {"role": "system", "content": "You are a helpful assistant that explains the veracity of a claim by considering the context. Instructions: - Only explain veracity of claims by considering just the related context."},
                        {"role": "user", "content": target_instance['prompt']}
                      ]
            
    return target_instances


  @backoff.on_exception(backoff.expo, RateLimitError)
  def __openai_query(self, prompt):
    ''' This function send a query to GPT-3.

    :param prompt: The target prompt
    :type prompt: str

    :returns: The generated text
    :rtype: str
    '''

    openai.api_key= OAI_API_KEY
    response = openai.Completion.create(engine= self.plms["gpt3"]["engine"] if self.plm_engine=='' else self.plm_engine
      , prompt= prompt, temperature= self.temperature , max_tokens= self.max_tokens
      , top_p=1, frequency_penalty=0
      , presence_penalty=0)
    
    return response.choices[0].text


  @backoff.on_exception(backoff.expo, RateLimitError)
  def __openai_chat_query(self, prompt):
    ''' This function send a query to open ai for using ChatGPT.

    :param prompt: The target prompt
    :type prompt: str

    :returns: The generated message
    :rtype: str
    '''

    openai.api_key= OAI_API_KEY
    response = openai.ChatCompletion.create(model= self.plms["chat_gpt"]["engine"] if self.plm_engine=='' else self.plm_engine
    , messages= prompt, temperature= self.temperature , max_tokens= self.max_tokens, top_p=1
    , frequency_penalty=0, presence_penalty=0)
    
    return response.choices[0].message.content


  def __hf_models_query(self, prompt):
    ''' This function send a query to HuggingFace models.

    :param prompt: The target prompt 
    :type prompt: str

    :returns: The generated text
    :rtype: str
    '''

    API_URL = f"https://api-inference.huggingface.co/models/{self.plms['gptj']['model_name']}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    parameters = {
        'max_new_tokens': self.max_tokens,  # number of generated tokens
        'temperature': self.temperature,   # controlling the randomness of generations
        'end_sequence': "###" # stopping sequence for generation
    }
    options={'use_cache': True}
    body = {"inputs":prompt,'parameters':parameters,'options':options}
    response = requests.post(API_URL, headers=headers, json= body)

    return response.json()[0]['generated_text']


  def __get_few_shot_demonstration(self, demonstration_instances, test_instances):
    ''' This function generates the few shot prompt including demonstration examples.

    :param demonstration_instances: The target instances to create the demonstration section of the prompt
    :type demonstration_instances: list
    :param test_instances: The target instances to infer by prompt paradigm
    :type test_instances: list    

    :returns: The input instance list with the few shot prompt including demonstration examples
    :rtype: list
    '''
    
    if self.selected_plm == "chat_gpt":
      # create demonestration section for Chat Completions
      demonstration_lst= [{"role": "system", "content": "You are a helpful assistant that explains the veracity of a claim by considering the context. Instructions: - Only explain veracity of claims by considering just the related context. - Use provided examples to learn more about explanation."}]
      for item in demonstration_instances:
        demonstration_lst.append({"role": "user", "content": self.prompt_template.format(item['summarized_main_text'], item['claim']
        , item['label'], "")})
        demonstration_lst.append({"role": "assistant", "content": item['explanation']})
      
      # create message with few shot section for Chat Completions
      for item in test_instances:
        item["prompt"]= demonstration_lst + [{"role": "user", "content": self.prompt_template.format(item['summarized_main_text'], item['claim']
        , item['label'], "")}]
      
      return test_instances

    # Create the demonstration section of the prompt
    self.__prompt(demonstration_instances, type="few")
    demonstration_str= ""
    for item in demonstration_instances:
      demonstration_str+= item["prompt"] + "###\n"

    # Create propmt for test instances. Add the demonstration section at the begining of each instances.
    self.__prompt(test_instances, type="zero")
    for item in test_instances:
      item["prompt"]= demonstration_str + item["prompt"]
    
    return test_instances


  def __check_plm(func):
    
    '''
    This is a decorator to check the selected PLM for the functions.
    '''
    def wrapper(self, *args, **kwargs):

      assert self.selected_plm in self.plms.keys(), f"Please select one of {self.plms.keys()} as PLM."

      func_result = func(self, *args, **kwargs)
      return func_result

    return wrapper


  @__check_plm
  def zero_shot(self, target_instances):
    ''' This function explain veracity of instances by using zero-shot and selected plm.

    :param target_instances: The input text 
    :type target_instances: list

    :returns: An input instance list with generated explanation added to each instance
    :rtype: list
    '''
    
    # create appropriate prompt
    self.__prompt(target_instances, type="zero")
    
    total_instances= len(target_instances)
    for index, target_instance in enumerate(target_instances):
      print(f"Generating explanation for {index+1}/{total_instances} ...")
      target_instance[self.selected_plm]= self.plms[self.selected_plm]["api_func"](target_instance['prompt'])
      
            
    return target_instances


  @__check_plm
  def few_shot(self, demonstration_instances, test_instances):
    ''' This function explain veracity of instances by using few-shot learning.

    :param demonstration_instances: The target instances to create the demonstration section of the prompt
    :type demonstration_instances: list
    :param test_instances: The target instances to infer by prompt paradigm
    :type test_instances: list

    :returns: An input instance list with generated explanation added to each instance
    :rtype: list
    '''

    # Create the the prompt for in-context learning
    self.__get_few_shot_demonstration(demonstration_instances, test_instances)

    # Call the API
    total_instances= len(test_instances)
    for index, target_instance in enumerate(test_instances):
      print(f"Generating explanation for {index+1}/{total_instances} ...")      
      target_instance[self.selected_plm]= self.plms[self.selected_plm]["api_func"](target_instance['prompt'])
    
    return test_instances