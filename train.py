import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import  all_tasks, read_df
from config import OLID_PATH , LABEL_DICT
from cli import get_args
from utils import load
from datasets import HuggingfaceDataset, HuggingfaceMTDataset, ImbalancedDatasetSampler
from models.bert import BERT, RoBERTa , MARBERT
from models.gated import GatedModel
from models.mtl import MTL_Transformer_LSTM
from models.mtl_noheads import MTL_Transformer_No_Heads
from models.mtl_cnn_lstm import MTL_Transformer_CNN_LSTM
from transformers import BertTokenizer, RobertaTokenizer, get_cosine_schedule_with_warmup
from trainer import Trainer
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModel

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
        
    # define the marbert
    if model_name == 'marbert':
        if task == 'all':
            # initialize the marbert MTL model
            if( args['modeltype'] == 'LSTM' ):
                print("Initialize MarBert with LSTM heads  ..... \n")
                model = MTL_Transformer_LSTM( model_name , model_size , args=args)
                
            elif( args['modeltype'] == 'NO_HEADS'  ):
                print("Initialize MarBert with No Heads ....... \n")
                model =  MTL_Transformer_No_Heads( model_name , model_size  , args= args )
                
            elif( args['modeltype'] == 'CNN_LSTM'  ):
                print("Initialize MarBert with CNN LSTM heads ....... \n")
                model =  MTL_Transformer_CNN_LSTM( model_name , model_size  , args= args )
            
        else:
            # should not run this , this is for normal bert classifier
            model = MARBERT( model_size , args=args, num_labels=num_labels )
            
        tokenizer = AutoTokenizer.from_pretrained( 'UBC-NLP/MARBERT')

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
        
    # final model evaluation on test dataset
    (labels_all_A, y_pred_all_A) , (labels_all_B, y_pred_all_B) , (labels_all_C, y_pred_all_C) , (labels_all_D, y_pred_all_D) ,(labels_all_E, y_pred_all_E) = trainer.test_m(stage='test')
    
    # define the classification report for every task
    print('\n')
    print("Sentiment Performance Analysis")
    print(classification_report( labels_all_A , y_pred_all_A , target_names= list(LABEL_DICT['a'].keys()) , zero_division = 0 ) )
    print('\n')
    print("Annotator Sentiment Performance Analysis")
    print(classification_report( labels_all_B , y_pred_all_B , target_names= list(LABEL_DICT['b'].keys()) , zero_division = 0 ) )
    print('\n')
    print("Directness Performance Analysis")
    print(classification_report( labels_all_C , y_pred_all_C , target_names= list(LABEL_DICT['c'].keys()) , zero_division = 0 ) )
    print('\n')
    print("Group Performance Analysis")
    print(classification_report( labels_all_D , y_pred_all_D , target_names= list(LABEL_DICT['d'].keys()) , zero_division = 0 ) )
    print('\n')
    print("Target Performance Analysis")
    print(classification_report( labels_all_E , y_pred_all_E , target_names= list(LABEL_DICT['e'].keys()) , zero_division = 0 ) )
    
