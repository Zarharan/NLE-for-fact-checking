from common.utils import *
import json
from openai.error import RateLimitError
import backoff
from data.pubhealth.models import *


class NLEGeneration():
  '''
  By using NLEGeneration object we can generate explanation.
  
  :param prompt_key: The key for the target template to create the prompt
  :type prompt_key: str
  :param temperature: To set the randomness of generated text (between 0 and 1, with 0 being the most predictable and 1 being the most random)
  :type temperature: float
  :param max_tokens: The maximum number of tokens for generated text
  :type max_tokens: int
  :param plm: The target pre-trained language model for few/zero shot learning
  :type plm: str  

  :ivar plms: List of possible pre-trained language models to select
  :vartype plms: list
  :ivar prompt_template: The target template to create the prompt
  :vartype prompt_template: str

  '''

  def __init__(self, prompt_key, plm="gpt3", temperature= 0.2, max_tokens= 200, plm_engine= ''):
    self.temperature= temperature
    self.max_tokens= max_tokens
    self.prompt_key= prompt_key
    template= prompt_key.split("/")
    self.prompt_template= PROMPT_TEMPLATES['PubHealth'][template[0]][template[1]]
    # if a specific template exists
    if prompt_key in CHAT_COMPLETION_SYSTEM_ROLE.keys():
      self.chat_completion_system_role_content= CHAT_COMPLETION_SYSTEM_ROLE[prompt_key]
    else:
      # use default template
      self.chat_completion_system_role_content= CHAT_COMPLETION_SYSTEM_ROLE[template[0]]
    self.selected_plm= plm
    self.plm_engine= plm_engine

    self._target_model= None 
    self._target_tokenizer= None
    self._device= torch.device('cpu')
    if torch.cuda.is_available():
      self._device = torch.device('cuda')
    
    self.plms= {"gpt3": {"api_func": self.__openai_query, "engine": "text-davinci-003", "zero_prompt_func": self.__prompt, "few_prompt_func": self.__general_few_shot_structure}
      , "chat_gpt": {"api_func": self.__openai_chat_query, "engine": "gpt-3.5-turbo", "zero_prompt_func": self.__chat_based_zero_shot_structure, "few_prompt_func": self.__chat_based_few_shot_structure}
      , "gpt4": {"api_func": self.__openai_chat_query, "engine": "gpt-4", "zero_prompt_func": self.__chat_based_zero_shot_structure, "few_prompt_func": self.__chat_based_few_shot_structure}
      , "vicuna": {"api_func": self.__open_source_instruction_based_lm, "engine": "lmsys/vicuna-13b-v1.5", "zero_prompt_func": self.__prompt, "few_prompt_func": self.__general_few_shot_structure}
      , "mistral": {"api_func": self.__open_source_instruction_based_lm, "engine": "mistralai/Mistral-7B-v0.1", "zero_prompt_func": self.__prompt, "few_prompt_func": self.__general_few_shot_structure}
      , "gptj": {"api_func": self.__hf_models_query, "model_name": "EleutherAI/gpt-j-6B", "zero_prompt_func": self.__prompt, "few_prompt_func": self.__general_few_shot_structure}}

    self._cgat_based_models= ["chat_gpt", "gpt4"]


  def __prompt(self, target_instance, type="zero"):
    ''' This function creates appropriate prompt for the input instance.

    :param target_instance: The input instance to create prompt for them
    :type target_instance: dict
    :param type: By using zero, the prompt does not include the gold explanation
    :type type: str

    :returns: The input instance with prompts added to it
    :rtype: dict
    '''
    target_instance['prompt'] = self.prompt_template.format("" if "bias_checking" in self.prompt_key else target_instance['summarized_main_text']
      , target_instance['claim']
      , "" if type=="zero" and any(item in self.prompt_key for item in ["veracity", "joint"]) else target_instance['label']
      , "" if type=="zero" or "veracity" in self.prompt_key else target_instance['explanation'])
            
    return target_instance


  def __general_few_shot_structure(self, demonstration_instances, test_instance):
    ''' This function generates the few shot structure including demonstration examples.

    :param demonstration_instances: The target instances to create the demonstration section of the prompt
    :type demonstration_instances: list
    :param test_instance: The target instance to infer by prompt paradigm
    :type test_instance: dict    

    :returns: The input instance with the few shot prompt including demonstration examples
    :rtype: dict
    '''    
    # Create the demonstration section of the prompt    
    demonstration_str= ""
    for item in demonstration_instances:
      demonstration_str+= self.__prompt(item, type="few")["prompt"] + "###\n"

    # Create propmt for test instances. Add the demonstration section at the begining of each instances.
    test_instance["prompt"]= demonstration_str + self.__prompt(test_instance, type="zero")["prompt"]
    
    return test_instance


  def __chat_based_zero_shot_structure(self, target_instance):
    ''' This function creates appropriate prompt for the input instance.

    :param target_instance: The input instance to create prompt for them
    :type target_instance: dict

    :returns: The input instance with prompts added to it
    :rtype: dict
    '''    
    if self.selected_plm in self._cgat_based_models:
      # create instruction section for Chat Completions
      instruction= [{"role": "system", "content": self.chat_completion_system_role_content}]
      
      target_instance["prompt"]= instruction + [{"role": "user", "content": self.__prompt(target_instance, type="zero")["prompt"]}]
      
      return target_instance
    else:
      raise Exception("Implement the appropriate structure for the selected PLM!")


  def __chat_based_few_shot_structure(self, demonstration_instances, test_instance):
    ''' This function generates the few shot structure including demonstration examples.

    :param demonstration_instances: The target instances to create the demonstration section of the prompt
    :type demonstration_instances: list
    :param test_instance: The target instance to infer by prompt paradigm
    :type test_instance: dict    

    :returns: The input instance with the few shot prompt including demonstration examples
    :rtype: dict
    '''
    if self.selected_plm in self._cgat_based_models:
      # create demonestration section for Chat Completions
      few_shot_additional_instruction= " - Use the provided examples to learn more about explanation."
      demonstration_lst= [{"role": "system", "content": self.chat_completion_system_role_content + few_shot_additional_instruction}]

      for item in demonstration_instances:
        demonstration_lst.append({"role": "user", "content": self.__prompt(item, type="zero")["prompt"]})
        reply= ""
        if "veracity" in self.prompt_key:
          reply= item['label']
        elif "explanation" in self.prompt_key:
          reply= item['explanation']
        else:
          reply= item['label'] + "\n" + item['explanation']          
        demonstration_lst.append({"role": "assistant", "content": reply})
      
      # Create propmt for test instances. Add the demonstration section at the begining of each instances.
      test_instance["prompt"]= demonstration_lst + [{"role": "user", "content": self.__prompt(test_instance, type="zero")["prompt"]}]
      
      return test_instance
    else:
      raise Exception("Implement the appropriate structure for the selected PLM!")


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


  def __open_source_instruction_based_lm(self, prompt):
    ''' This function generate a text as response by using the model, tokenizer, and given prompt

    :param prompt: The target prompt 
    :type prompt: str

    :returns: The generated text
    :rtype: str
    '''

    if self._target_model is None or self._target_tokenizer is None: 
      self._target_model = AutoModelForCausalLM.from_pretrained(self.plms[self.selected_plm]["engine"])
      self._target_tokenizer = LlamaTokenizer.from_pretrained(self.plms[self.selected_plm]["engine"])
      self._target_model = self._target_model.to(self._device)

    inputs = self._target_tokenizer (prompt, return_tensors="pt")

    outputs = self._target_model.generate(input_ids=sample["input_ids"].to(self._device), max_new_tokens=self.max_tokens, temperature= self.temperature)
    return self._target_tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]


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
  def zero_shot(self, target_instances, experiment_id):
    ''' This function explain veracity of instances by using zero-shot and selected plm.

    :param target_instances: The input text 
    :type target_instances: list
    :param experiment_id: The Id of the experiment that we want to save instances results for 
    :type experiment_id: int

    :returns: An input instance list with generated explanation added to each instance
    :rtype: list
    '''    
    experiments= Experiments()

    total_instances= len(target_instances)
    for index, target_instance in enumerate(target_instances):
      claim_id= target_instance["claim_id"]
      instance_result= experiments.select_instance_result(experiment_id, claim_id)
      if instance_result:
        log(f"The claim with Id {claim_id} was ignored because its result exists.")
        target_instance[self.selected_plm]= instance_result.result
        continue

      log(f"Generating explanation for {index+1}/{total_instances} ...")
      # create appropriate prompt
      self.plms[self.selected_plm]["zero_prompt_func"](target_instance)
      target_instance[self.selected_plm]= self.plms[self.selected_plm]["api_func"](target_instance['prompt'])
      
      # save each instance result into the DB
      instance_data= ExperimentInstancesModel(experiment_id= experiment_id
        , claim_id= claim_id, result= target_instance[self.selected_plm])
      experiments.insert_instances(instance_data)
            
    return target_instances


  @__check_plm
  def few_shot(self, demonstration_instances, test_instances, experiment_id):
    ''' This function explain veracity of instances by using few-shot learning.

    :param demonstration_instances: The target instances to create the demonstration section of the prompt
    :type demonstration_instances: list
    :param test_instances: The target instances to infer by prompt paradigm
    :type test_instances: list
    :param experiment_id: The Id of the experiment that we want to save instances results for 
    :type experiment_id: int    

    :returns: An input instance list with generated explanation added to each instance
    :rtype: list
    '''
    experiments= Experiments()

    # Call the API
    total_instances= len(test_instances)
    for index, target_instance in enumerate(test_instances):
      claim_id= target_instance["claim_id"]
      instance_result= experiments.select_instance_result(experiment_id, claim_id)
      if instance_result:
        log(f"The claim with Id {claim_id} was ignored because its result exists.")
        target_instance[self.selected_plm]= instance_result.result
        continue

      log(f"Generating explanation for {index+1}/{total_instances} ...")      
      # create appropriate prompt
      self.plms[self.selected_plm]["few_prompt_func"](demonstration_instances, target_instance)
      target_instance[self.selected_plm]= self.plms[self.selected_plm]["api_func"](target_instance['prompt'])

      # save each instance result into the DB
      instance_data= ExperimentInstancesModel(experiment_id= experiment_id
        , claim_id= claim_id, result= target_instance[self.selected_plm])
      experiments.insert_instances(instance_data)
    
    return test_instances