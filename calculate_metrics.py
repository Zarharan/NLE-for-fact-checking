from common.utils import *
import argparse
import pandas as pd
from pathlib import Path


def report_metrics(file_path, pred_col_title, target_col_title, metrics_obj):

    path = Path(file_path)
    assert path.is_file(), f"Please enter a correct path to a csv file."
    target_file_df= pd.read_csv(file_path)

    print(target_file_df[pred_col_title].tolist())
    print(target_file_df[target_col_title].tolist())
    
    metrics_obj.pred_list = target_file_df[pred_col_title].tolist()
    metrics_obj.target_list = target_file_df[target_col_title].tolist()

    print(metrics_obj.get_all_metrics())


def main():
    # Initialize parser
    parser = argparse.ArgumentParser()
       
    parser.add_argument("-pred_col_title", "--pred_col_title", help = "The title of the prediction column to report metrics for it"
        , default='gpt3', type= str)
    parser.add_argument("-target_col_title", "--target_col_title", help = "The title of target (ground truth) column to report metrics regarding it"
        , default='explanation', type= str)
    parser.add_argument("-type", "--type"
        , help = "Report metrics for a single file or all files in a folder", default="folder"
        , choices=['file', 'folder'], type= str)
    parser.add_argument("-target_path", "--target_path"
        , help = "The path to the file or folder to report metrics for them", default="data/pubhealth/prompts/"
        , type= str)
    parser.add_argument("-bertscore_model", "--bertscore_model"
        , help = " A name or a model path used to load transformers pretrained model to calculate the Bert score"
        , default="microsoft/deberta-xlarge-mnli"
        , type= str)        

    # Read arguments from command line
    args = parser.parse_args()    
    
    nle_metrics= NLEMetrics(bertscore_model= args.bertscore_model)

    if args.type == "file":
        report_metrics(args.target_path, args.pred_col_title, args.target_col_title, nle_metrics)
    else:
        # ToDo: complete this section later
        pass


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here

    main()