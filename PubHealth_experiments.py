from common.dataset import PubHealthDataset
from common.nle_generation import NLEGeneration
from common.utils import *
import argparse
import pandas as pd

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here

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

    # parser.add_argument("-test_instances_no", "--test_instances_no", help = "The number of test instances"
    # , default=1, type= int)

    parser.add_argument("-seed", "--seed", help = "seed for random function. Pass None for select different instances randomly."
    , default=313, type= int)
    parser.add_argument("-prompt_template", "--prompt_template", help = "The target template to create prompt"
    , default='explanation/basic', choices=['explanation/basic', 'veracity/basic'])
    parser.add_argument("-prompt_type", "--prompt_type", help = "zero shot or few shot"
    , default='zero', choices=['zero', 'few'])
    

    # Read arguments from command line
    args = parser.parse_args()
    
    assert args.train_path, "At least enter the train set path!"

    # Object for summarization the main text of the news
    summarization= Summarization()

    # create the dataset object to read examples
    pubhealth_dataset= PubHealthDataset(train_path= args.train_path, val_path= args.val_path, test_path= args.test_path)

    # create NLEGeneration object and create zero or few shot prompts
    nle_generator = NLEGeneration()
    template= args.prompt_template.split("/")

    if args.prompt_type == "zero":
        selected_instances= pubhealth_dataset.get_k_rand_instances(k_per_class= args.k_per_class
            , k_rand_instance=args.k_rand_instance, target_set= args.test_target_set
            , random_seed= args.seed, summarization_method= SUMMARIZATION_KEY_VAL[args.summarize])
        
        nle_generator.gpt3_zero_shot(selected_instances, PROMPT_TEMPLATES['PubHealth'][template[0]][template[1]])

        gpt3_zero_shot_df = pd.DataFrame(selected_instances)
        # save results in a csv file
        gpt3_zero_shot_df.to_csv(f"data/pubhealth/prompts/gpt3_zero_shot_{args.seed}.csv")


    # View result
    print("selected_instances length:", len(selected_instances))
    for item in selected_instances:
        print("-" * 50)
        print(item["prompt"])
        print("-"*20)
        print(item['result'])

    
#  A sample of few shot inference with summarization by using gpt3 and select from test set
# python PubHealth_experiments.py -summarize gpt3 -k_per_class 1 -k_rand_instance 0 -test_path data/pubhealth/test.tsv