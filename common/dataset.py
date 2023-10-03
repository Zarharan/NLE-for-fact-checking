import numpy as np
from pathlib import Path
import pandas as pd
from common.preprocessor import Preprocessor
# import random


class PubHealthDataset():
  '''
  The PubHealthDataset object contains different types of dataset for example: datafram, list of tokens, or list of cleaned text, etc.
  :param train_path: The path of train set
  :type train_path: str
  :param val_path: The path of validation set
  :type val_path: str
  :param test_path: The path of test set
  :type test_path: str

  :ivar df_orginal_trainset: a pandas datafram which includes original train set
  :vartype df_orginal_trainset: datafram
  :ivar df_orginal_testset: a pandas datafram which includes original test set
  :vartype df_orginal_testset: datafram
  :ivar df_orginal_valset: a pandas datafram which includes original validation set
  :vartype df_orginal_valset: datafram
  :ivar label_space: list of all labels in the dataset
  :vartype label_space: list
  :ivar pre_processor: an object for different pre processing tasks
  :vartype pre_processor: object
  :ivar all_available_sets: list of different section of the dataset (train, val, test)
  :vartype all_available_sets: list
  '''

  def __init__(self, train_path, val_path="", test_path= ""):    

    self.df_orginal_trainset= self.read_dataset(train_path, 'train')
    self.df_orginal_testset= None
    self.df_orginal_valset= None

    if val_path is not None and len(val_path)>0:
      self.df_orginal_valset = self.read_dataset(val_path, 'validation')
    
    if test_path is not None and len(test_path)>0:
      self.df_orginal_testset = self.read_dataset(test_path, 'test')

    self.label_space= ['unproven','true','false','mixture']
    self.pre_processor= Preprocessor()
    self.all_available_sets= ['train', 'val', 'test']


  def read_dataset(self, data_path, set_title):
    ''' This function gets the path and title of dataset (tsv) and read it into dataframe.
    
    :param data_path: The path of tsv file
    :type text: str
    :param set_title: The title of target set. Acceptable values are train, val, and test.
    :type set_title: str

    :returns: pandas dataframe
    :rtype: object
    '''

    assert set_title in ['train', 'val', 'test'], f"Acceptable values for set_title are {all_available_sets}."
    path = Path(data_path)
    assert path.is_file(), f"Please enter the correct path to the {set_title} set."

    temp_set= pd.read_table(data_path)
    if set_title== "train":
      self.df_orginal_trainset= temp_set
    elif set_title== "val":
      self.df_orginal_valset= temp_set
    else:
      self.df_orginal_testset= temp_set

    return temp_set


  def get_k_rand_instances(self, k_per_class= 2, k_rand_instance=1, target_set= 'train'
    , random_seed= 313, summarization_obj= None, exclude_claim_ids= None):
    ''' This function selects k1 instances per label and k2 instances regardless of class randomly and cleans them by using pre_processor object.
    
    :param k_per_class: The number of random samples per class.
    :type k_per_class: int
    :param k_rand_instance: The number of random samples regardless of class.
    :type k_rand_instance: int    
    :param target_set: The target set to select from. Acceptable values are train, val, and test.
    :type target_set: str
    :param random_seed: seed for random function. Pass None for select different instances randomly.
    :type random_seed: int
    :param summarization_obj: The object to summarize the main text of the news
    :type summarization_obj: function
    :param exclude_claim_ids: You can exclude instances by passing related claim IDs if you selected them before and don't want to select them again.
    :type exclude_claim_ids: list or series    

    :returns: List of Cleaned examples
    :rtype: list
    '''

    assert target_set in ['train', 'val', 'test'], f"Acceptable values for target_set are {all_available_sets}."

    if target_set== "train":
      assert self.df_orginal_trainset is not None, "Please read the train set at first!"
      temp_df= self.df_orginal_trainset.loc[self.df_orginal_trainset['label'].isin(self.label_space)]
    elif target_set== "val":
      assert self.df_orginal_valset is not None, "Please read the validation set at first!"
      temp_df= self.df_orginal_valset.loc[self.df_orginal_valset['label'].isin(self.label_space)]
    else:
      assert self.df_orginal_testset is not None, "Please read the test set at first!"
      temp_df= self.df_orginal_testset.loc[self.df_orginal_testset['label'].isin(self.label_space)]

    # Exclude claim Ids that we do not want to select them
    if exclude_claim_ids is not None:
      temp_df= temp_df[~temp_df["claim_id"].isin(exclude_claim_ids)]

    np.random.seed(random_seed)
    rand_instances_df= None
    self.k_rand_clean_examples= []

    # select k random instances per class
    if k_per_class> 0:
      fn = lambda obj: obj.loc[np.random.choice(obj.index, k_per_class, False),:]

      rand_instances_df= temp_df.groupby('label', as_index=False).apply(fn)
      rand_instances_df= rand_instances_df.sample(frac=1)

    # select k random instances regardless of class
    if k_rand_instance>0:
      if k_per_class> 0:
        rand_instances_df= pd.concat([temp_df[~temp_df["claim_id"].isin(rand_instances_df['claim_id'])].sample(n=k_rand_instance, random_state= random_seed)
          ,rand_instances_df])
      else:
        rand_instances_df= temp_df.sample(n=k_rand_instance, random_state= random_seed)    

    for index, row in rand_instances_df.iterrows():
      main_text= self.pre_processor.clean_text(row['main_text'])
      self.k_rand_clean_examples.append({"claim_id":row['claim_id']
          ,"claim":self.pre_processor.clean_text(row['claim'])
          ,"main_text":main_text,"label":row['label']
          , "explanation":self.pre_processor.clean_text(row['explanation'])
          , "summarized_main_text": main_text})  

    print(f"The instances were read successfully from {target_set}.")

    if summarization_obj:
      for sample in self.k_rand_clean_examples:
        sample["summarized_main_text"]= summarization_obj.get_summary(sample["main_text"], sample["claim_id"], sample["claim"])

      print(f"Successfully summarized the main text of the instances read from {target_set}.")

    return self.k_rand_clean_examples