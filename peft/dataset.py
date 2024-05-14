from functools import partial
from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd


PROMPT_TEMPLATES = {
    "veracity" : ("""### Context: {0}\nClaim: {1}\n\n### Class:\n{2}"""),
    "explanation": ("""### Context: {0}\nClaim: {1}\nThe claim veracity: {2}\n\n### Explanation:\n{3}"""),
    "joint":("""### Context: {0}\nClaim: {1}\n\n### Response:\n{2}{3}"""),
    
}


def read_dataset(file_path, task_type="veracity", validate_mode= False):
    """
    Reading the target dataset method

    :param file_path: The dataset file path
    :param task_type: Explanation, veracity prediction, or joint task
    :param validate_mode: Prepare dataset with prompts for evaluating or training
    """
    target_file_df= pd.read_csv(file_path)
    target_file_df= target_file_df.filter(items=['claim_id', 'claim', 'summarized_text', 'label', 'explanation'])

    # Remove instances that include null values
    target_file_df = target_file_df.dropna()

    lst_texts= []
    lst_labels= []
    # label_2_id = {'false': 0, 'true': 1, 'mixture': 2, 'unproven': 3}

    # Add prompt to all rows
    for index, row in target_file_df.iterrows():
        lst_texts.append(create_prompt_formats(row, task_type, validate_mode))
        # lst_labels.append(label_2_id[row["label"].lower()])
        lst_labels.append(str(row["label"]).lower())

    target_file_df["text"]= lst_texts
    target_file_df["labels"]= lst_labels

    dataset= Dataset.from_pandas(target_file_df)
    print(f'Number of prompts: {len(dataset)}')
    print(f'Column names are: {dataset.column_names}')

    return dataset


def create_prompt_formats(sample, task_type="veracity", validate_mode= False):
    """
    Creates a formatted prompt template for a prompt in the instruction dataset

    :param sample: Prompt or sample from the instruction dataset
    :param task_type: Explanation, veracity prediction, or joint task
    :param validate_mode: Prepare the prompts for evaluating or training    
    """

    label= sample['label']
    explanation= sample['explanation']
    if validate_mode and task_type in ["veracity", "joint"]:
      label= ""
    
    if validate_mode or task_type == "veracity":
      explanation= ""

    if not validate_mode and task_type == "joint":
      label= "{\"veracity\":\"" + sample['label'] + "\","
      explanation= "\"explanation\":\"" + sample['explanation'] + "\"}"

    return PROMPT_TEMPLATES[task_type].format(sample['summarized_text'], sample['claim'], label, explanation)


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizes dataset batch

    :param batch: Dataset batch
    :param tokenizer: Model tokenizer
    :param max_length: Maximum number of tokens to emit from the tokenizer
    """

    return tokenizer(
        batch["text"],
        max_length = max_length,
        truncation = True,
    )


def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str, remove_columns:list):
    """
    Tokenizes dataset for fine-tuning

    :param tokenizer (AutoTokenizer): Model tokenizer
    :param max_length (int): Maximum number of tokens to emit from the tokenizer
    :param seed: Random seed for reproducibility
    :param dataset (str): Instruction dataset
    """

    # Add prompt to each sample
    print("Preprocessing dataset...")

    # Apply preprocessing to each batch of the dataset & and remove "instruction", "input", "output", and "text" fields
    _preprocessing_function = partial(preprocess_batch, max_length = max_length, tokenizer = tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched = True,
        remove_columns = remove_columns
    )

    # Filter out samples that have "input_ids" exceeding "max_length"
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    print("----------- instances after filtering out long samples: ", len(dataset))

    # Shuffle dataset
    dataset = dataset.shuffle(seed = seed)

    return dataset
