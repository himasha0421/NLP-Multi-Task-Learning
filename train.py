import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import task_a, task_b, task_c, all_tasks, read_df
from config import OLID_PATH
from cli import get_args
from utils import load
from datasets import HuggingfaceDataset, HuggingfaceMTDataset, ImbalancedDatasetSampler
from models.bert import BERT, RoBERTa
from models.gated import GatedModel
from models.mtl import MTL_Transformer_LSTM
from transformers import BertTokenizer, RobertaTokenizer, get_cosine_schedule_with_warmup
from trainer import Trainer

TRAIN_PATH = "arab_dataset/arab_trainset.csv"
TEST_PATH = "arab_dataset/arab_testset.csv" 
VALID_PATH = "arab_dataset/arab_validset.csv"

if __name__ == '__main__':
    # Get command line arguments
    args = get_args()
    task = args['task']
    model_name = args['model']
    model_size = args['model_size']
    truncate = args['truncate']
    epochs = args['epochs']
    lr = args['learning_rate']
    wd = args['weight_decay']
    bs = args['batch_size']
    patience = args['patience']

    # Fix seed for reproducibility
    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_labels = 3 if task == 'c' else 2

    # Set tokenizer for different models
    if model_name == 'bert':
        if task == 'all':
            model = MTL_Transformer_LSTM(model_name, model_size, args=args)
        else:
            model = BERT(model_size, args=args, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(f'bert-{model_size}-uncased')
    elif model_name == 'roberta':
        if task == 'all':
            model = MTL_Transformer_LSTM(model_name, model_size, args=args)
        else:
            model = RoBERTa(model_size, args=args, num_labels=num_labels)
        tokenizer = RobertaTokenizer.from_pretrained(f'roberta-{model_size}')
    elif model_name == 'bert-gate' and task == 'all':
        model_name = model_name.replace('-gate', '')
        model = GatedModel(model_name, model_size, args=args)
        tokenizer = BertTokenizer.from_pretrained(f'bert-{model_size}-uncased')
    elif model_name == 'roberta-gate' and task == 'all':
        model_name = model_name.replace('-gate', '')
        model = GatedModel(model_name, model_size, args=args)
        tokenizer = RobertaTokenizer.from_pretrained(f'roberta-{model_size}')

    # Move model to correct device
    model = model.to(device=device)

    if args['ckpt'] != '':
        model.load_state_dict(load(args['ckpt']))

    # Read in data depends on different subtasks
    # label_orders = {'a': ['OFF', 'NOT'], 'b': ['TIN', 'UNT'], 'c': ['IND', 'GRP', 'OTH']}

    if task in ['all']:
        # read the train , test and valid datasets
        token_ids, lens , mask , label_a, label_b, label_c , label_d , label_e = all_tasks(TRAIN_PATH, tokenizer=tokenizer, truncate=truncate)
        test_token_ids, test_lens, test_mask , test_label_a, test_label_b, test_label_c , test_label_d , test_label_e = all_tasks( TEST_PATH , tokenizer=tokenizer, truncate=truncate)
        valid_token_ids, valid_lens, valid_mask , valid_label_a, valid_label_b, valid_label_c , valid_label_d , valid_label_e = all_tasks( VALID_PATH , tokenizer=tokenizer, truncate=truncate)
        
        labels = {'a': label_a, 'b': label_b, 'c': label_c , 'd': label_d ,'e': label_e }
        test_labels = {'a': test_label_a, 'b': test_label_b, 'c': test_label_c , 'd': test_label_d , 'e': test_label_e }
        valid_labels = {'a': valid_label_a, 'b': valid_label_b, 'c': valid_label_c , 'd': valid_label_d ,'e': valid_label_e }
        
        _Dataset = HuggingfaceMTDataset

    datasets = {
        'train': _Dataset(
            input_ids=token_ids,
            lens=lens,
            mask=mask,
            labels=labels,
            task=task
        ),
        'test': _Dataset(
            input_ids=test_token_ids,
            lens=test_lens,
            mask=test_mask,
            labels=test_labels,
            task=task
        ),
        'valid': _Dataset(
            input_ids= valid_token_ids,
            lens= valid_lens,
            mask= valid_mask,
            labels= valid_labels,
            task=task
        )
    }

    sampler = ImbalancedDatasetSampler(datasets['train']) if task in ['a', 'b', 'c'] else None
    
    dataloaders = {
        'train': DataLoader(
            dataset=datasets['train'],
            batch_size=bs,
            sampler=sampler
        ),
        'test': DataLoader(dataset=datasets['test'], batch_size=bs) ,
        'valid': DataLoader(dataset=datasets['valid'], batch_size=bs) ,
    }

    criterion = torch.nn.CrossEntropyLoss()

    if args['scheduler']:
        optimizer = torch.optim.AdamW( model.parameters() , lr=lr, weight_decay=wd)
        # A warmup scheduler
        t_total = epochs * len(dataloaders['train'])
        warmup_steps = np.ceil(t_total / 10.0) * 2
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_total
        )
    else:
        optimizer = torch.optim.Adam( model.parameters() , lr=lr, weight_decay=wd)
        scheduler = None

    trainer = Trainer(
        model=model,
        epochs=epochs,
        dataloaders=dataloaders,
        criterion=criterion,
        loss_weights=args['loss_weights'],
        clip=args['clip'],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=patience,
        task_name=task,
        model_name=model_name,
        seed=args['seed']
    )

    if task in ['a', 'b', 'c']:
        trainer.train()
    else:
        trainer.train_m()
