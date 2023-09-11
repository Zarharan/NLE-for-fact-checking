import argparse
from model.gpt_trainer import *
from model.dataset import PubHealthDataset
from model.generation_utils import *
from torch.utils.data import DataLoader
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr",default=5e-5, type=float, help="learning rate")
    parser.add_argument("-seed",default=313, type=int, help="seed to replicate results")
    parser.add_argument("-n_gpu",default=1, type=int, help="no of gpu available")
    parser.add_argument("-gradient_accumulation_steps",default=4, type=int, help="gradient_accumulation_steps")    
    parser.add_argument("-num_workers",default=4, type=int, help="num of cpus available")
    parser.add_argument("-device",default=torch.device('cpu'), help="torch.device object")
    parser.add_argument("-num_train_epochs",default=5, type=int, help="no of epochs of training")
    parser.add_argument("-output_dir",default='output', type=str, help="path to save evaluation results")
    parser.add_argument("-model_dir",default='weights', type=str, help="path to save trained model")
    parser.add_argument("-fp16",default=True, type=bool, help="whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("-fp16_opt_level",default='O0', type=str, help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
    parser.add_argument("-max_grad_norm",default=1.0, type=float, help="max gradient norm.")
    parser.add_argument("-train_path", help = "The tsv file path of the train set"
    , default='data/pubhealth/train.tsv', type= str)
    parser.add_argument("-val_path", default='data/pubhealth/dev.tsv', help = "The tsv file path of the validation set")
    parser.add_argument("-test_path", default='data/pubhealth/test.tsv', help = "The tsv file path of the test set")
    parser.add_argument("-truncation",default=True, type=bool, help="set truncation for the input text of the model")
    parser.add_argument("-main_text_max_length",default=1000, type=int, help="Set the max length for the main text section")
    parser.add_argument("-explanation_max_length",default=1000, type=int, help="Set the max length for the explanation section")
    parser.add_argument("-batch_size",default=1, type=int, help="batch_size")
    parser.add_argument("-tokenizer", default='EleutherAI/gpt-j-6b', help = "The name of target tokenizer")
    parser.add_argument("-model_name", default="EleutherAI/gpt-j-6b", help = "The name of target model")
    parser.add_argument("-top_p",default=0.5, type=float, help="top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).")
    parser.add_argument("-top_k",default=10, type=int, help="top_k > 0: keep only top k tokens with highest probability (top-k filtering).")
    parser.add_argument("-temperature",default=0.8, type=float, help="temperature >0: used to control the randomness of predictions by scaling the logits before applying softmax.")
    parser.add_argument("-generation_length",default=70, type=int, help="length of generated sequence")           

    args = parser.parse_args()

    # Get the tokenizer with special tokens
    tokenizer = tokenizer_with_special_tokens(args.tokenizer)

    # Read the trining set and load into dataloader
    train_dataset = PubHealthDataset(data_path= args.train_path, tokenizer=tokenizer
        , truncation= args.truncation, main_text_max_length= args.main_text_max_length
        , explanation_max_length= args.explanation_max_length, pre_processor= None)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    log("The training Dataloader length= " + str(len(train_dataloader.dataset)))

    # Read the validation set and load into dataloader
    val_dataset = PubHealthDataset(data_path= args.val_path, tokenizer=tokenizer
        , truncation= args.truncation, main_text_max_length= args.main_text_max_length
        , explanation_max_length= args.explanation_max_length, pre_processor= None)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    log("The validation Dataloader length= " + str(len(val_dataloader.dataset)))
    
    # Load the model into the device
    ignore_idx = tokenizer.pad_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    start = time.time()

    # Train the model
    train(args, model, tokenizer, train_dataloader, val_dataloader, val_dataset, ignore_idx)
    log('total time: ', (time.time()-start)/60, " minutes", '\n\n')

    log('Saving trained model...')
    
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    model_file = os.path.join(args.model_dir, 'model_{}_data{}_trained_after_{}_epochs_only_sum_loss_ignr_pad.bin'.format(args.fp16_opt_level,3000,args.num_train_epochs))
    config_file = os.path.join(args.model_dir, 'config_{}_data{}_trained_after_{}_epochs_only_sum_loss_ignr_pad.json'.format(args.fp16_opt_level,3000,args.num_train_epochs))
    torch.save(model.state_dict(), model_file)
    model.config.to_json_file(config_file)


if __name__ == '__main__':
	main()
