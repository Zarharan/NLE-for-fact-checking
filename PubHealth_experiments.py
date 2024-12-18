from common.dataset import PubHealthDataset
from common.nle_generation import NLEGeneration
from common.utils import *
import argparse
import pandas as pd
from pathlib import Path
from dict_hash import sha256
from data.pubhealth.models import *


# logging
file_handler = logging.FileHandler('PubHealth_experiments.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# End of logging


def clean_explanation(explanation):
    '''
    This function clean explanation and remove all extra text after ###

    :param explanation: The input explanation
    :type explanation: str

    :returns: cleaned explanation
    :rtype: str
    '''

    explanation= " ".join([line for line in explanation.split("\n") if len(line.strip())>1])
    explanation= explanation.strip("\"\"").strip("\": \"").strip("''").strip("': '")
    return explanation.split("###")[0]


def seperate_veracity_from_explanation(input_text):
    '''
    This function extracts and returns the veracity label and explanation from input text for joint task

    :param input_text: The input text
    :type input_text: str

    :returns: Extracted veracity label and explanation
    :rtype: tuple
    '''

    input_text= " ".join([line for line in input_text.split("\n") if len(line.strip())>1])
    if input_text.strip() == "":
      return "other", ""

    # remove the curly braces from the string
    input_text = input_text.strip('{}').replace("Veracity", "veracity", 1).replace("Explanation", "explanation", 1).replace("explanation", "###EOVS###", 1)

    responses = input_text.split("###EOVS###")
    veracity= responses[0].split(":")
    if len(veracity)==1:
      veracity= veracity[0]
    elif len(veracity)==2:
      veracity= veracity[1]
    else:
      veracity = veracity[1] if "veracity" in veracity[0] else veracity[2]
      veracity = veracity.split("explanation")[0]
    veracity = veracity.split()[0].replace(",","").replace("'","").replace("\"","")

    return veracity.lower(), clean_explanation(responses[1])


def main():
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Adding optional argument
    parser.add_argument("-train_path", "--train_path", help = "The tsv file path of the train set"
    , default='data/pubhealth/train.tsv', type= str)
    parser.add_argument("-val_path", "--val_path", help = "The tsv file path of the validation set")
    parser.add_argument("-test_path", "--test_path", help = "The tsv file path of the test set")
    parser.add_argument("-k_per_class", "--k_per_class", help = "The number of samples per class for test section"
    , default=4, type= int)
    parser.add_argument("-k_rand_instance", "--k_rand_instance", help = "The number of random samples regardless of class for test section"
    , default=4, type= int)

    parser.add_argument("-demon_k_per_class", "--demon_k_per_class", help = "The number of samples per class for demonstration section"
    , default=0, type= int)
    parser.add_argument("-demon_k_rand_instance", "--demon_k_rand_instance", help = "The number of random samples regardless of class for demonstration section"
    , default=1, type= int)

    parser.add_argument("-demon_target_set", "--demon_target_set"
    , help = "The target set to select the demonstration instances from"
    , default='train', choices=['train', 'val', 'test'])
    parser.add_argument("-test_target_set", "--test_target_set"
    , help = "The target set to select the test instances from"
    , default='test', choices=['train', 'val', 'test'])
    parser.add_argument("-summarize", "--summarize"
    , help = "Whether summarize the main text of the news or not"
    , default='false', choices=['false', 'gpt3', 'bart', 'lsg_bart', 'chatgpt'])
    parser.add_argument("-summarization_model_path", "--summarization_model_path"
    , help = "The path of weights of the target model to generate summary (except api-based models)"
    , default='data/models/lsg_bart', type=str)
    parser.add_argument("-summarization_max_token", "--summarization_max_token", help = "The max number of tokens for generated summary."
    , default=200, type= int)
    parser.add_argument("-summarization_temperature", "--summarization_temperature", help = "To set the randomness of generated summary."
    , default=0.5, type= float)
    parser.add_argument("-seed", "--seed", help = "seed for random function. Pass None for select different instances randomly."
    , default=313, type= int)
    parser.add_argument("-prompt_template", "--prompt_template", help = "The target template to create prompt"
    , default='explanation/claude_suggestion', choices=['explanation/basic', 'veracity/basic', 'explanation/natural', 'veracity/natural', 'joint/basic', 'joint/natural', 'explanation/bias_checking', 'veracity/bias_checking', 'joint/bias_checking', 'explanation/claude_suggestion', 'veracity/claude_suggestion', 'joint/claude_suggestion', 'veracity/instruction_based', 'explanation/instruction_based', 'joint/instruction_based'])
    parser.add_argument("-prompt_type", "--prompt_type", help = "zero shot or few shot"
    , default='zero', choices=['zero', 'few'])
    parser.add_argument("-plm", "--plm", help = "gpt3, chat_gpt(gpt-3.5-turbo), gpt4, vicuna, mistral, or gptj"
    , default='gpt3', choices=['gpt3','chat_gpt', 'gpt4','gptj', 'vicuna', 'mistral', 'falcon', 'llama'])
    parser.add_argument("-plm_engine", "--plm_engine", help = "For chat completion: gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301. And for completion: text-davinci-003, text-davinci-002, text-curie-001, text-babbage-001, text-ada-001"
    , default='', choices=['','gpt-4', 'gpt-4-0314', 'gpt-4-32k', 'gpt-4-32k-0314', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0301', 'text-davinci-003', 'text-davinci-002', 'text-curie-001', 'text-babbage-001', 'text-ada-001', 'mistralai/Mistral-7B-v0.1', 'lmsys/vicuna-13b-v1.5', 'mistralai/Mistral-7B-Instruct-v0.1', 'lmsys/vicuna-33b-v1.3', 'tiiuae/falcon-40b', 'meta-llama/Llama-2-13b-hf'])
    parser.add_argument("-nle_temperature", "--nle_temperature", help = "To set the randomness of generated explanation."
    , default=0.5, type= float)
    parser.add_argument("-add_chatgpt_prompt", "--add_chatgpt_prompt", help = "Add another coloumn to the result file for ChatGPT prompt."
    , default=False, type= bool)
    parser.add_argument("-explanation_max_token", "--explanation_max_token", help = "The max number of tokens for generated explanation."
    , default=300, type= int)    

    # Read arguments from command line
    args = parser.parse_args()

    assert args.train_path, "At least enter the train set path!"

    # create NLEGeneration object and create zero or few shot prompts
    
    nle_generator = NLEGeneration(max_tokens= args.explanation_max_token, prompt_key=args.prompt_template
        , temperature= args.nle_temperature, plm=args.plm,plm_engine= args.plm_engine)
    
    # number of instances to test. 4 is number of labels in this task.
    instances_no = 4 * args.k_per_class + args.k_rand_instance   
    k_rand_instance_no= args.k_rand_instance if args.prompt_type=="zero" else args.demon_k_rand_instance
    k_per_class_no= args.k_per_class if args.prompt_type=="zero" else args.demon_k_per_class
    # File name to save the results of the experiment for the selected configuration
    save_path= "data/pubhealth/prompts/" + args.prompt_template + "/"
    result_file_name= f"{save_path}{nle_generator.selected_plm}_{args.prompt_type}_{k_per_class_no}_{k_rand_instance_no}_{instances_no}_{args.seed}.csv"
    
    # Check whether the results of the experiment for the selected configuration exists in DB or not.
    args_dict= vars(args)
    hashed_args_dict= sha256(args_dict)

    experiments= Experiments()
    experiment_existence= experiments.select_experiment(hashed_args_dict)
    target_experiment_id= None
    
    if experiment_existence: # When the experiment exists
        target_experiment_id= experiment_existence.id

        if any(experiment_existence.results): 
            # When the experiment includes different save file(s) for the result and it means it is already completed
            log(f"You have already did experiment(s) with entered arguments.\nEntered arguments values:\n{args_dict} \nRelated file result(s): \n")
            for result in experiment_existence.results:
                log(f"{result.file_path}\n")
            
            log("Do you want to continue?")
            input_command = input('Enter c to continue, any other key to cancel and exit ... ').strip()[0]
            if input_command != "c" and input_command != "C":
                log("The experiment was canceled!")
                return
            
            log("Write a new name for the result file or press enter key to continue with a default name.")
            input_file_name = input().strip()

            if len(input_file_name)== 0:
                log("The new results will be saved with a default name!")
                result_file_name= result_file_name.replace(".csv", f"_{get_utc_time()}.csv")
            else:            
                result_file_name= f"{save_path}{input_file_name}.csv"
            
            log(f"The new file name is: {result_file_name}")

    else: # When the experiment does not exist
        # save all arguments of the experiment
        experiment_data= ExperimentModel(args= args_dict, args_hash= hashed_args_dict, completed= False)
        experiments.insert(experiment_data)
        target_experiment_id= experiment_data.id

    # Object for summarization the main text of the news
    summarization= None
    if args.summarize != "false":
        summarization= Summarization(max_tokens= args.summarization_max_token
        , temperature= args.summarization_temperature, model_name= args.summarize
        , model_path= args.summarization_model_path)

    # create the dataset object to read examples
    pubhealth_dataset= PubHealthDataset(train_path= args.train_path, val_path= args.val_path, test_path= args.test_path)

    nle_result= []
    if args.prompt_type == "zero":
        selected_instances= pubhealth_dataset.get_k_rand_instances(k_per_class= args.k_per_class
            , k_rand_instance=args.k_rand_instance, target_set= args.test_target_set
            , random_seed= args.seed, summarization_obj= summarization)
        
        nle_result= nle_generator.zero_shot(selected_instances, target_experiment_id)
    
    elif args.prompt_type == "few":

        demonstration_instances= pubhealth_dataset.get_k_rand_instances(k_per_class= args.demon_k_per_class
            , k_rand_instance=args.demon_k_rand_instance, target_set= args.demon_target_set
            , random_seed= args.seed, summarization_obj= summarization)

        test_instances= pubhealth_dataset.get_k_rand_instances(k_per_class= args.k_per_class
            , k_rand_instance=args.k_rand_instance, target_set= args.test_target_set
            , random_seed= args.seed, summarization_obj= summarization
            , exclude_claim_ids= pd.DataFrame(demonstration_instances)['claim_id'])
        
        nle_result= nle_generator.few_shot(demonstration_instances, test_instances, target_experiment_id)

    if args.add_chatgpt_prompt:
        add_chatgpt_prompt(nle_result, args.prompt_type)

    # Seperate veracity from explanation in joint task
    if "joint" in args.prompt_template:
        for target_instance in nle_result:
            veracity, explanation = seperate_veracity_from_explanation(target_instance[args.plm])
            target_instance[args.plm+"_veracity"]= veracity
            target_instance[args.plm+"_explanation"]= explanation

    elif "explanation" in args.prompt_template and args.plm in OPEN_SOURCED_MODELS:
        for target_instance in nle_result:
            cleaned_explanation = clean_explanation(target_instance[args.plm])
            target_instance[args.plm+"_cleaned"]= cleaned_explanation
        

    # save results in a csv file
    Path(save_path).mkdir(parents=True, exist_ok=True)

    nle_result_df = pd.DataFrame(nle_result)    
    nle_result_df.to_csv(result_file_name)

    # Save the result file path into DB
    # Only add new result since the arguments have been already saved
    experiment_result= ExperimentResultModel(experiment_id= target_experiment_id,file_path = result_file_name)
    experiments.insert_result(experiment_result)
    # update the completion of the experiment
    experiments.update_completion(experiment_id= target_experiment_id)

    log(f"The experiment Done! the result was saved at {result_file_name}")


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    try:
        main()
    except Exception as err:
        log(f"Unexpected error: {err}, type: {type(err)}")


#  A sample of few shot inference with summarization by using gpt3
# and select four samples (one per class) for demonestration section of the prompt
# python PubHealth_experiments.py -demon_k_per_class 1 -demon_k_rand_instance 0 -k_per_class 4 -k_rand_instance 4 -test_path data/pubhealth/test.tsv -summarize bart -prompt_type few -summarization_max_token 300