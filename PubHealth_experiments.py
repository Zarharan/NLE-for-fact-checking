from common.dataset import PubHealthDataset
from common.nle_generation import NLEGeneration
from common.utils import *
import argparse
import pandas as pd
from pathlib import Path


def main():
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Adding optional argument
    parser.add_argument("-train_path", "--train_path", help = "The tsv file path of the train set"
    , default='data/pubhealth/train.tsv', type= str)
    parser.add_argument("-val_path", "--val_path", help = "The tsv file path of the validation set")
    parser.add_argument("-test_path", "--test_path", help = "The tsv file path of the test set")
    parser.add_argument("-k_per_class", "--k_per_class", help = "The number of samples per class for demonstration"
    , default=0, type= int)
    parser.add_argument("-k_rand_instance", "--k_rand_instance", help = "The number of random samples regardless of class for demonstration"
    , default=1, type= int)
    parser.add_argument("-demon_target_set", "--demon_target_set"
    , help = "The target set to select the demonstration instances from"
    , default='train', choices=['train', 'val', 'test'])
    parser.add_argument("-test_target_set", "--test_target_set"
    , help = "The target set to select the test instances from"
    , default='test', choices=['train', 'val', 'test'])
    parser.add_argument("-summarize", "--summarize"
    , help = "Whether summarize the main text of the news or not"
    , default='false', choices=['false', 'gpt3', 'bart'])
    parser.add_argument("-summarization_max_token", "--summarization_max_token", help = "The max number of tokens for generated summary."
    , default=200, type= int)    
    parser.add_argument("-seed", "--seed", help = "seed for random function. Pass None for select different instances randomly."
    , default=313, type= int)
    parser.add_argument("-prompt_template", "--prompt_template", help = "The target template to create prompt"
    , default='explanation/basic', choices=['explanation/basic', 'veracity/basic'])
    parser.add_argument("-prompt_type", "--prompt_type", help = "zero shot or few shot"
    , default='zero', choices=['zero', 'few'])
    parser.add_argument("-test_instances_no", "--test_instances_no", help = "The number of test instances"
    , default=20, type= int)
    

    # Read arguments from command line
    args = parser.parse_args()    
    assert args.train_path, "At least enter the train set path!"

    # create NLEGeneration object and create zero or few shot prompts
    template= args.prompt_template.split("/")
    nle_generator = NLEGeneration(PROMPT_TEMPLATES['PubHealth'][template[0]][template[1]])
    
    # number of instances to test for few shot
    instances_no= args.test_instances_no
    if args.prompt_type == "zero": # number of instances to test for zero shot
        instances_no = 4 * args.k_per_class + args.k_rand_instance
    
    # File name to save the results of the experiment for the selected configuration
    save_path= "data/pubhealth/prompts/"
    result_file_name= f"{save_path}{nle_generator.selected_plm}_{args.prompt_type}_{args.k_per_class}_{args.k_rand_instance}_{instances_no}_{args.seed}.csv"
    
    # Check whether the file containing the results of the experiment for the selected configuration exists or not.
    path = Path(result_file_name)    
    if path.is_file():
        print("There is a file containing the results of the experiment for the selected configuration!")
        print("The file name: ",result_file_name)
        print("Do you want to continue?")
        input_command = input('Enter c to continue, any other key to cancel and exit ... ').strip()[0]
        if input_command != "c" and input_command != "C":
            print("The experiment was canceled!")
            return
        
        print("Write a new name for the result file or press enter key to continue and replace the new result with the existing one.")
        input_file_name = input().strip()

        if len(input_file_name)== 0:
            print("The new results will be replaced with the existing one!")
        else:            
            result_file_name= f"{save_path}{input_file_name}.csv"
            print(f"new file name is: {result_file_name}")

    # Object for summarization the main text of the news
    summarization= Summarization()

    # create the dataset object to read examples
    pubhealth_dataset= PubHealthDataset(train_path= args.train_path, val_path= args.val_path, test_path= args.test_path)

    propmt_result= []

    if args.prompt_type == "zero":
        selected_instances= pubhealth_dataset.get_k_rand_instances(k_per_class= args.k_per_class
            , k_rand_instance=args.k_rand_instance, target_set= args.test_target_set
            , random_seed= args.seed, summarization_method= SUMMARIZATION_KEY_VAL[args.summarize]
            , summarization_max_token= args.summarization_max_token)
        
        nle_generator.zero_shot(selected_instances)

        gpt3_zero_shot_df = pd.DataFrame(selected_instances)
        # save results in a csv file
        gpt3_zero_shot_df.to_csv(result_file_name)
        propmt_result= selected_instances
    
    elif args.prompt_type == "few":

        demonstration_instances= pubhealth_dataset.get_k_rand_instances(k_per_class= args.k_per_class
            , k_rand_instance=args.k_rand_instance, target_set= args.demon_target_set
            , random_seed= args.seed, summarization_method= SUMMARIZATION_KEY_VAL[args.summarize]
            , summarization_max_token= args.summarization_max_token)

        test_instances= pubhealth_dataset.get_k_rand_instances(k_per_class= 0
            , k_rand_instance=args.test_instances_no, target_set= args.test_target_set
            , random_seed= args.seed, summarization_method= SUMMARIZATION_KEY_VAL[args.summarize]
            , summarization_max_token= args.summarization_max_token
            , exclude_claim_ids= pd.DataFrame(demonstration_instances)['claim_id'])
        
        nle_generator.few_shot(demonstration_instances, test_instances)

        gpt3_few_shot_df = pd.DataFrame(test_instances)
        # save results in a csv file
        gpt3_few_shot_df.to_csv(result_file_name)
        
        propmt_result= test_instances

    print(f"The experiment Done! the result was saved at {result_file_name}")


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here

    main()

    
#  A sample of zero shot inference with summarization by using gpt3 and select from test set
# python PubHealth_experiments.py -summarize bart -k_per_class 1 -k_rand_instance 0 -test_path data/pubhealth/test.tsv

#  A sample of few shot inference with summarization by using gpt3
# and select four samples (one per class) for demonestration section of the prompt
# python PubHealth_experiments.py -k_per_class 1 -k_rand_instance 0 -test_path data/pubhealth/test.tsv -summarize bart -prompt_type few -summarization_max_token 300