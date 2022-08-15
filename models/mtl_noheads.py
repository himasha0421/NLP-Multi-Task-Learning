import torch
from torch import nn
from transformers import BertModel, RobertaModel
from .modules.attention import Attention
from transformers import AutoTokenizer, AutoModel

class MTL_Transformer_No_Heads(nn.Module):
    def __init__(self, model, model_size, args ):
        super( MTL_Transformer_No_Heads , self ).__init__()
        input_size = 768 if model_size == 'base' else 1024

        if model== 'marbert':
            MODEL = AutoModel
            model_full_name = 'UBC-NLP/MARBERT'

        self.emb = MODEL.from_pretrained(
            model_full_name,
            hidden_dropout_prob=args['hidden_dropout'],
            attention_probs_dropout_prob=args['attention_dropout']
        )

        
        self.dropout = nn.Dropout(p=args['dropout'])
        
        self.Linears = nn.ModuleDict({
            'a': nn.Sequential(
                nn.Linear( input_size , 6)
            ),
            'b': nn.Sequential(
                nn.Linear( input_size , 7)
            ),
            'c': nn.Sequential(
                nn.Linear( input_size , 2)
            ),
            'd': nn.Sequential(
                nn.Linear( input_size , 4)
            ),
            'e': nn.Sequential(
                nn.Linear( input_size , 4)
            )
        })

    def forward(self, inputs, lens, mask):
        """
        Bert Model Outputs :
              1.  last_hidden_state 
              2.  pooler_output
              3.  hidden_states
              4.  attentions
        """
        # get the sequence prediction from the marbert , ( batch size , hidden size (768) )
        martbert_embeds = self.emb(inputs, attention_mask=mask)[1] # (batch_size, hidden_size)
        
        # apply dropout to avoid model overfitting
        embeds_dropout = self.dropout( martbert_embeds )
        ####################################################################

        logits_a = self.Linears['a']( embeds_dropout )
        logits_b = self.Linears['b']( embeds_dropout )
        logits_c = self.Linears['c']( embeds_dropout )
        logits_d = self.Linears['d']( embeds_dropout )
        logits_e = self.Linears['e']( embeds_dropout )

        return logits_a, logits_b, logits_c ,logits_d , logits_e
