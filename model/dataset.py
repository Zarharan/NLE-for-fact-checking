from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from common.utils import *
import torch


class PubHealthDataset(Dataset):
    '''
    The PubHealthDataset object contains different types of dataset for example: datafram, list of tokens, or list of cleaned text, etc.
    :param data_path: The path of target dataset
    :type data_path: str
    :param tokenizer: The tokenizer object
    :type tokenizer: object
    :param truncation: Truncate long main text and explanations or not
    :type truncation: boolean
    :param main_text_max_length: The max length for the claim and main text pairs
    :type main_text_max_length: int
    :param explanation_max_length: The max length for the explanation
    :type explanation_max_length: int
    :param pre_processor: The object for preprocessing. It has to include a function with the name clean_text.
    :type pre_processor: object

    :ivar original_df: a pandas datafram which includes original dataset
    :vartype original_df: datafram
    :ivar claims: The list of all claims in the dataset
    :vartype claims: list
    :ivar main_texts: The list of all main texts in the dataset
    :vartype main_texts: list
    :ivar veracity_labels: The list of all veracity labels in the dataset
    :vartype veracity_labels: list
    :ivar explanations: The list of all explanations in the dataset
    :vartype explanations: list
    :ivar _model_input: The dictionary that contains the tokenized inputs (claim + [sep] main_text + [sep] + explanation) for all instances in the dataset.
    :vartype _model_input: dict
    '''
    def __init__(self, data_path, tokenizer, truncation, main_text_max_length, explanation_max_length, pre_processor= None):
        self.data_path= data_path
        self.tokenizer = tokenizer
        self.tokenizer.sep_token= '<|sep|>'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.truncation = truncation
        self.main_text_max_length= main_text_max_length
        self.explanation_max_length = explanation_max_length
        self.pre_processor = pre_processor

        self.__read_dataset()
        self.__prepare_features()


    def __read_dataset(self):
        ''' This function reads the dataset and loads it into a dataframe. In addition it does preprocessing if the related object has been assigned.
        
        :returns: Nothing
        :rtype: None
        '''

        path = Path(self.data_path)
        assert path.is_file(), f"Please enter the correct path to the target dataset."
        
        log("Reading dataset from: " + self.data_path)

        if ".csv" in self.data_path:
            self.original_df = pd.read_csv(self.data_path)
        elif ".xlsx" in self.data_path:
            self.original_df = pd.read_excel(self.data_path)
        elif ".tsv" in self.data_path:
            self.original_df= pd.read_table(self.data_path)
        else:
            raise Exception("Implement an appropriate reading method!")

        # Remove instances that include null values
        self.original_df = self.original_df.dropna()

        self.claims = self.original_df['claim']
        self.claim_ids =self.original_df['claim_id']
        self.main_texts = self.original_df['main_text']
        self.veracity_labels = self.original_df['label']

        # The number of explanation tokens in the majority of instances is less than 1000 tokens (in the train and dev set)
        self.explanations = self.original_df['explanation']

        # Log the class distribution of veracity
        log("Veracity labels distribution: \n", self.original_df.groupby(['label'])['label'].count())
        self._dataset_size= len(self.claims)
        log(f"The dataset shape: {self.original_df.shape}")
        
        # Do preprocessing if the related object has been assigned
        if self.pre_processor is not None:
            self.claims = self.claims.apply(self.pre_processor.clean_text)
            self.main_texts = self.main_texts.apply(self.pre_processor.clean_text)
            self.explanations = self.explanations.apply(self.pre_processor.clean_text)
            log('The dataset was cleaned!')

        log('Reading the dataset done!')


    def __prepare_features(self):
        ''' This function tokenize all instances and saves the result into _model_input property.
        
        :returns: Nothing
        :rtype: None
        '''

        log("Preparing features ...")

        self._model_input = {'input_ids':[], 'attention_mask':[], 'target_len':[] }
        for claim, main_text, label, explanation in zip(self.claims.tolist(), self.main_texts.tolist(), self.veracity_labels.tolist(), self.explanations.tolist()):
            main_text_tokenized= self.tokenizer(claim + self.tokenizer.pad_token + main_text, return_tensors="pt", truncation=self.truncation
                , max_length= self.main_text_max_length, padding="max_length", add_special_tokens=True)
            main_text_len= min(len(claim + self.tokenizer.pad_token + main_text), self.main_text_max_length)
            explanation_tokenized= self.tokenizer(self.tokenizer.pad_token + label + self.tokenizer.pad_token + explanation, return_tensors="pt", truncation=self.truncation
                , max_length= self.explanation_max_length, padding="max_length", add_special_tokens=True)
            context_len= min(len(claim + self.tokenizer.pad_token + main_text), self.main_text_max_length)
            self._model_input["input_ids"].append(torch.concat((main_text_tokenized["input_ids"], explanation_tokenized["input_ids"]), 1))
            self._model_input["target_len"].append(context_len)
            self._model_input["attention_mask"].append(torch.concat((main_text_tokenized["attention_mask"], explanation_tokenized["attention_mask"]), 1))

        log('Preparing features done!')


    def __getitem__(self, idx):
        ''' The function returns the related features of the instance at the input index.
        
        :param idx: The index of the target instance
        :type idx: int

        :returns: A dictionary that consists of features of the related instance
        :rtype: dict
        '''

        target_output = {'input_ids':self._model_input["input_ids"][idx]
            ,'target_len':self._model_input["target_len"][idx], 'attention_mask':self._model_input["attention_mask"][idx]}
        return target_output


    def __len__(self):
        ''' The function returns the length of target dataset.

        :returns: The length of target dataset
        :rtype: int
        '''        
        return self._dataset_size