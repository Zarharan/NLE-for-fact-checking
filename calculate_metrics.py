from common.utils import *
import argparse
import pandas as pd
from pathlib import Path
import glob
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


# logging
file_handler = logging.FileHandler('calculate_metrics.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# End of logging


def get_pred_target_colms(file_path, *args):
    ''' This function read a file and returns the content of the selected columns in args.

    :param file_path: The path to the file to report metrics for it
    :type file_path: str
    :param *args: The title of target columns to get their contents as a list
    :type *args: str

    :returns: The content of selected columns
    :rtype: tuple
    '''

    path = Path(file_path)
    assert path.is_file(), f"Please enter a correct path to a csv file."
    target_file_df= pd.read_csv(file_path)
    target_file_df= target_file_df.fillna('')
    lst_result=[]
    
    for arg in args:
        lst_result.append(target_file_df[arg].tolist())

    return tuple(lst_result)


def report_nle_metrics(file_path, pred_col_title, target_col_title, metrics_obj, target_metric):
    ''' This function read a file and report evaluation metrics for the generated explanation in the file.

    :param file_path: The path to the file to report metrics for it
    :type file_path: str
    :param pred_col_title: The title of the prediction column to report metrics for it
    :type pred_col_title: str
    :param target_col_title: The title of target (ground truth) column to report metrics regarding it
    :type target_col_title: str
    :param metrics_obj: An instance of NLEMetrics class to obtain metrics
    :type metrics_obj: object
    :param target_metric: The target metric you want to calculate
    :type target_metric: str

    :returns: The calculated metrics
    :rtype: dict
    '''
    
    metrics_obj.pred_list, metrics_obj.target_list, metrics_obj.claim_list, metrics_obj.claim_gold_label_list = get_pred_target_colms(file_path, pred_col_title, target_col_title, "claim", "label")

    metric_fun_map = {'all': metrics_obj.get_all_metrics
                        , 'rouge': metrics_obj.rouge_score
                        , 'SGC': metrics_obj.SGC
                        , 'WGC': metrics_obj.WGC
                        , 'LC': metrics_obj.LC
                        , 'bleu': metrics_obj.bleu_score}

    target_metric_result= metric_fun_map[target_metric]()
    log(target_metric_result)
    
    return target_metric_result


def report_veracity_metrics(file_path, pred_col_title, target_col_title, veracity_average_method):
    ''' This function read a file and report evaluation metrics for veracity classification task.

    :param file_path: The path to the file to report metrics for it
    :type file_path: str
    :param pred_col_title: The title of the prediction column to report metrics for it
    :type pred_col_title: str
    :param target_col_title: The title of target (ground truth) column to report metrics regarding it
    :type target_col_title: str
    :param veracity_average_method: The average method fot reporting recal, precision, and F1.
    :type veracity_average_method: str
    
    :returns: The calculated metrics
    :rtype: dict
    '''

    pred_list, target_list = get_pred_target_colms(file_path, pred_col_title, target_col_title)

    def _clean_generated_labels(label):
      label= label.strip().replace("#","").replace("\n","").lower().split()
      if len(label)>0:
        return label[0]
      return "other"

    pred_list = list(map(_clean_generated_labels, pred_list))
    labels=['true', 'false', 'mixture', 'unproven']

    metrics_result= {"acc": accuracy_score(y_pred=pred_list, y_true=target_list)
        , "pre": precision_score(y_pred=pred_list, y_true=target_list, average=veracity_average_method, labels=labels)
        , "rec": recall_score(y_pred=pred_list, y_true=target_list, average=veracity_average_method, labels=labels)
        , "f1": f1_score(y_pred=pred_list, y_true=target_list, average=veracity_average_method, labels=labels)
        , "confmat": confusion_matrix(y_pred=pred_list, y_true=target_list, labels=labels)}

    return metrics_result


def main():
    # Initialize parser
    parser = argparse.ArgumentParser()
       
    parser.add_argument("-pred_col_title", "--pred_col_title", help = "The title of the prediction column to report metrics for it"
        , default='gpt3', type= str)
    parser.add_argument("-target_col_title", "--target_col_title", help = "The title of target (ground truth) column to report metrics regarding it"
        , default='explanation', type= str)
    parser.add_argument("-type", "--type"
        , help = "Report metrics for a single file or all files in a directory", default="directory"
        , choices=['file', 'directory'], type= str)
    parser.add_argument("-target_path", "--target_path"
        , help = "The path to the file or directory to report metrics for them", default="data/pubhealth/prompts/"
        , type= str)
    parser.add_argument("-bertscore_model", "--bertscore_model"
        , help = " A name or a model path used to load transformers pretrained model to calculate the Bert score"
        , default="microsoft/deberta-xlarge-mnli"
        , type= str)
    parser.add_argument("-task_type", "--task_type"
        , help = "Report metrics for the veracity prediction, the explanation generation or the joint model", default="explanation"
        , choices=['explanation', 'veracity', 'joint'], type= str)
    parser.add_argument("-veracity_average_method", "--veracity_average_method"
        , help = "The average method fot reporting recal, precision, and F1.", default="macro"
        , choices=['macro', 'micro', 'weighted'], type= str)        
    parser.add_argument("-nli_model_name", "--nli_model_name"
        , help = " A model name to calculate the SGC and WGC scores", default="roberta_large_snli"
        , choices=['roberta_large_snli', 'allennlp_nli_models'], type= str)
    parser.add_argument("-nli_model_path", "--nli_model_path"
        , help = " The path of selected model to calculate the SGC and WGC scores. Different options for AllenNLP models: pair-classification-decomposable-attention-elmo, pair-classification-roberta-mnli, or pair-classification-roberta-snli"
        , default="data/models/roberta_large_snli", type= str)
    parser.add_argument("-target_metric", "--target_metric"
        , help = "The target metric you want to calculate. It is only for explanation!", default="all"
        , choices=['all', 'rouge', 'SGC', 'WGC', 'LC', 'bleu'], type= str)        

    # Read arguments from command line
    args = parser.parse_args()
    
    nle_metrics= NLEMetrics(bertscore_model= args.bertscore_model)
    nle_metrics.nli_model= NLI(args.nli_model_name, args.nli_model_path)

    files = []

    log("Input arguments for calculating metric(s): ", vars(args))

    if args.type == "file":
        files.append(args.target_path)
    else:
        # csv files in the path
        files = glob.glob(args.target_path + "/*.csv")

    for file_name in files:
        log("-"*50)
        log(f"Calculating metrics for {file_name}:")

        if args.task_type=="veracity":
            log(report_veracity_metrics(file_name, args.pred_col_title, args.target_col_title, args.veracity_average_method))
        elif args.task_type=="explanation":
            report_nle_metrics(file_name, args.pred_col_title, args.target_col_title, nle_metrics, args.target_metric)
        else: # joint task
            pass
        

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here

    try:
        main()
    except Exception as err:
        log(f"Unexpected error: {err}, type: {type(err)}")