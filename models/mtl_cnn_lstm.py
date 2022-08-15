import torch
from torch import nn
from transformers import BertModel, RobertaModel
from .modules.attention import Attention
from transformers import AutoTokenizer, AutoModel

# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 

class MTL_Transformer_CNN_LSTM(torch.nn.Module):
    def __init__(self , model, model_size, args ):
        super( MTL_Transformer_CNN_LSTM , self ).__init__()
        
        input_size = 768 if model_size == 'base' else 1024
        hidden_size = args['hidden_size']
        self.concat = args['hidden_combine_method'] == 'concat'
        
        if model== 'marbert':
            MODEL = AutoModel
            model_full_name = 'UBC-NLP/MARBERT'

        self.model = MODEL.from_pretrained(
            model_full_name,
            hidden_dropout_prob=args['hidden_dropout'],
            attention_probs_dropout_prob=args['attention_dropout']
        )


        # define the forward hook function
        self.activation = {}
        def get_activation(name):
              def hook(model, input, output):
                  self.activation[name] = output
              return hook

        # register the forward hook for each encoder block
        for i_encoder in range(  12 ):
          self.model.base_model.encoder.layer[i_encoder].output.dropout.register_forward_hook(get_activation('encoder_layer{}_out'.format( i_encoder )  ))
        
        # define the relu activation function
        self.relu = torch.nn.ReLU()
        # define the dropout layer
        self.dropout = nn.Dropout(p=args['dropout'])
        
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
        
        # define the convolution layer
        self.conv = nn.Conv2d( in_channels= 12 , out_channels= 1 , kernel_size=1 )
        
        linear_in_features = hidden_size * 2 if self.concat else hidden_size
        
        self.Linears = nn.ModuleDict({
            'a': nn.Sequential(
                nn.Linear(linear_in_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 6)
            ),
            'b': nn.Sequential(
                nn.Linear(linear_in_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 7)
            ),
            'c': nn.Sequential(
                nn.Linear(linear_in_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 2)
            ),
            'd': nn.Sequential(
                nn.Linear(linear_in_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 4)
            ),
            'e': nn.Sequential(
                nn.Linear(linear_in_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 4)
            )
        })
        

    
    def forward(self, inputs, lens, mask ):

        # forward pass the bert model
        base_output = self.model( inputs , attention_mask = mask )

        # get the intermediate activations from dense and concatenate
        stacked_vec = torch.stack( list( self.activation.values() ) , axis= 1  )

        # apply the 1D convolution
        conv_out =  self.relu( self.conv( stacked_vec ) )
        conv_out =  conv_out.squeeze(1)
        
        # sentiment score head
        _, (h_a, _) = self.LSTMs['a'](conv_out)
        if self.concat:
            h_a = torch.cat((h_a[0], h_a[1]), dim=1)
        else:
            h_a = h_a[0] + h_a[1]
        h_a = self.dropout(h_a)
        ####################################################################
        
        # annotator score head
        _, (h_b, _) = self.LSTMs['b'](conv_out)
        if self.concat:
            h_b = torch.cat((h_b[0], h_b[1]), dim=1)
        else:
            h_b = h_b[0] + h_b[1]
        h_b = self.dropout(h_b)
        ####################################################################
        
        # directness score head
        _, (h_c, _) = self.LSTMs['c'](conv_out)
        if self.concat:
            h_c = torch.cat((h_c[0], h_c[1]), dim=1)
        else:
            h_c = h_c[0] + h_c[1]
        h_c = self.dropout(h_c)
        ####################################################################
        
        # group score head
        _, (h_d, _) = self.LSTMs['d'](conv_out)
        if self.concat:
            h_d = torch.cat((h_d[0], h_d[1]), dim=1)
        else:
            h_d = h_d[0] + h_d[1]
        h_d = self.dropout(h_d)
        ####################################################################
        
        # target score head
        _, (h_e, _) = self.LSTMs['e'](conv_out)
        if self.concat:
            h_e = torch.cat((h_e[0], h_e[1]), dim=1)
        else:
            h_e = h_e[0] + h_e[1]
        h_e = self.dropout(h_e)
        ####################################################################
        
        # forward pass the head dense layers and activations 
        logits_a = self.Linears['a'](h_a)
        logits_b = self.Linears['b'](h_b)
        logits_c = self.Linears['c'](h_c)
        logits_d = self.Linears['d'](h_d)
        logits_e = self.Linears['e'](h_e)

        return logits_a, logits_b, logits_c ,logits_d , logits_e