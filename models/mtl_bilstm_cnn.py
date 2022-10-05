import torch
from torch import nn
from transformers import BertModel, RobertaModel
from .modules.attention import Attention
from transformers import AutoTokenizer, AutoModel

class MTL_Transformer_BiLSTM_CNN(nn.Module):
    def __init__(self, model, model_size, args):
        super(MTL_Transformer_BiLSTM_CNN , self).__init__()
        hidden_size = args['hidden_size']
        self.concat = args['hidden_combine_method'] == 'concat'
        input_size = 768 if model_size == 'base' else 1024

        if model== 'marbert':
            MODEL = AutoModel
            model_full_name = 'UBC-NLP/MARBERT'

        self.emb = MODEL.from_pretrained(
            model_full_name,
            hidden_dropout_prob=args['hidden_dropout'],
            attention_probs_dropout_prob=args['attention_dropout']
        )

        self.LSTMs = nn.ModuleDict({
            'a': nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=args['num_layers'],
                bidirectional=True,
                batch_first=True,
                dropout=args['dropout'] if args['num_layers'] > 1 else 0
            ),
            'b': nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=args['num_layers'],
                bidirectional=True,
                batch_first=True,
                dropout=args['dropout'] if args['num_layers'] > 1 else 0
            ),
            'c': nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=args['num_layers'],
                bidirectional=True,
                batch_first=True,
                dropout=args['dropout'] if args['num_layers'] > 1 else 0
            ),
            'd': nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=args['num_layers'],
                bidirectional=True,
                batch_first=True,
                dropout=args['dropout'] if args['num_layers'] > 1 else 0
            ),
            'e': nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=args['num_layers'],
                bidirectional=True,
                batch_first=True,
                dropout=args['dropout'] if args['num_layers'] > 1 else 0
            )
        })

        self.dropout = nn.Dropout(p=args['dropout'])

        cnn_in_channels , cnn_out_channels = 2 ,  1

        # define two new cnn layers with relu activation function
        self.cnn1d_layers =  nn.ModuleDict({
          'a':  nn.Sequential(
                  nn.Conv1d( cnn_in_channels , cnn_out_channels , kernel_size = 3 , padding = 1 , stride = 1) ,
                  nn.BatchNorm1d( cnn_out_channels ) ,
                  nn.ReLU(),
                  nn.Dropout(p=args['dropout'])
          ),
          'b':  nn.Sequential(
                  nn.Conv1d( cnn_in_channels , cnn_out_channels , kernel_size = 3 , padding = 1 , stride = 1) ,
                  nn.BatchNorm1d( cnn_out_channels ) ,
                  nn.ReLU(),
                  nn.Dropout(p=args['dropout'])
          ),
          'c':  nn.Sequential(
                  nn.Conv1d( cnn_in_channels , cnn_out_channels , kernel_size = 3 , padding = 1 , stride = 1) ,
                  nn.BatchNorm1d( cnn_out_channels ) ,
                  nn.ReLU(),
                  nn.Dropout(p=args['dropout'])
          ),
          'd':  nn.Sequential(
                  nn.Conv1d( cnn_in_channels , cnn_out_channels , kernel_size = 3 , padding = 1 , stride = 1) ,
                  nn.BatchNorm1d( cnn_out_channels ) ,
                  nn.ReLU(),
                  nn.Dropout(p=args['dropout'])
          ),
          'e':  nn.Sequential(
                  nn.Conv1d( cnn_in_channels , cnn_out_channels , kernel_size = 3 , padding = 1 , stride = 1) ,
                  nn.BatchNorm1d( cnn_out_channels ) ,
                  nn.ReLU(),
                  nn.Dropout(p=args['dropout'])
          ),

        })

        
        self.Linears = nn.ModuleDict({
            'a': nn.Sequential(
                nn.Linear(hidden_size, 6)
            ),
            'b': nn.Sequential(
                nn.Linear(hidden_size, 7)
            ),
            'c': nn.Sequential(
                nn.Linear(hidden_size, 2)
            ),
            'd': nn.Sequential(
                nn.Linear(hidden_size, 4)
            ),
            'e': nn.Sequential(
                nn.Linear(hidden_size, 4)
            )
        })

    def forward(self, inputs, lens, mask):
      
        embs = self.emb(inputs, attention_mask=mask)[0] # (batch_size, sequence_length, hidden_size)
        
        # sentiment score head
        out_a , (h_a, _) = self.LSTMs['a'](embs)

        # concat the final activation outputs
        h_a = torch.cat((h_a[0].unsqueeze( dim=1 ), h_a[1].unsqueeze( dim=1 )  ), dim=1)
        h_a = self.dropout(h_a)

        # apply convolution
        print(h_a.shape)
        h_a  =  self.cnn1d_layers['a']( h_a ).squeeze(dim=1)


        ####################################################################
        
        # annotator score head
        out_b , (h_b, _) = self.LSTMs['b'](embs)

        # apply hidden state concatenation
        h_b = torch.cat((h_b[0].unsqueeze( dim=1 )  , h_b[1].unsqueeze( dim=1 ) ), dim=1)
        h_b = self.dropout(h_b)

        # apply convolution
        h_b  =  self.cnn1d_layers['b']( h_b ).squeeze(dim=1)

        ####################################################################
        
        # directness score head
        out_c , (h_c, _) = self.LSTMs['c'](embs)

        # apply hidden state concatenation
        h_c = torch.cat((h_c[0].unsqueeze( dim=1 ) , h_c[1].unsqueeze( dim=1 ) ), dim=1)
        h_c = self.dropout(h_c)

        # apply convolution
        h_c  =  self.cnn1d_layers['c']( h_c ).squeeze(dim=1)

        ####################################################################
        
        # group score head
        out_d , (h_d, _) = self.LSTMs['d'](embs)
        
        # apply hidden state concatenation
        h_d = torch.cat((h_d[0].unsqueeze( dim=1 ) , h_d[1].unsqueeze( dim=1 ) ), dim=1)
        h_d = self.dropout(h_d)

        # apply convolution
        h_d  =  self.cnn1d_layers['d']( h_d ).squeeze(dim=1)

        ####################################################################
        
        # target score head
        out_e , (h_e, _) = self.LSTMs['e'](embs)

        # apply hidden state concatenation
        h_e = torch.cat((h_e[0].unsqueeze( dim=1 ) , h_e[1].unsqueeze( dim=1 ) ), dim=1)
        h_e = self.dropout(h_e)

        # apply convolution
        h_e  =  self.cnn1d_layers['e']( h_e ).squeeze(dim=1)

        ####################################################################

        logits_a = self.Linears['a'](h_a)
        logits_b = self.Linears['b'](h_b)
        logits_c = self.Linears['c'](h_c)
        logits_d = self.Linears['d'](h_d)
        logits_e = self.Linears['e'](h_e)

        return logits_a, logits_b, logits_c ,logits_d , logits_e
