import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, AutoModelForSequenceClassification


def save_bart(model_name, save_path):
    ''' This function saves the target transformer in the path.

    :param model_name: The model name of bart transformer
    :type model_name: str
    :param save_path: The target save path
    :type save_path: str    

    :returns: None
    '''    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"The model ({model_name}) was saved successfully in {save_path} .")


def save_lsg_bart(model_name, save_path):
    ''' This function saves the lsg bart transformer in the path.

    :param model_name: The model name of lsg bart transformer
    :type model_name: str
    :param save_path: The target save path
    :type save_path: str    

    :returns: None
    '''    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"The model ({model_name}) was saved successfully in {save_path} .")    


def save_sequence_classification(model_name, save_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"The model ({model_name}) was saved successfully in {save_path} .")


def main():
    parser = argparse.ArgumentParser()
       
    parser.add_argument("-save_path", "--save_path", help = "The path to save the target transformer."
        , default='data/models/bart', type= str)
    parser.add_argument("-transformer_fn", "--transformer_fn"
        , help = "The name of function to save a transformer.", default="bart"
        , choices=['bart', 'lsg_bart', 'sequence_classification'], type= str)
    parser.add_argument("-model_name", "--model_name"
        , help = "The model name of target transformer to save.", default="philschmid/bart-large-cnn-samsum"
        , type= str)

    # Read arguments from command line
    args = parser.parse_args()

    if args.transformer_fn== "bart":
        save_bart(args.model_name, args.save_path)
    elif args.transformer_fn== "lsg_bart":
        save_lsg_bart(args.model_name, args.save_path)
    elif args.transformer_fn== "sequence_classification":
        save_sequence_classification(args.model_name, args.save_path)        
    # Add other transformers if needed!


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here

    main()