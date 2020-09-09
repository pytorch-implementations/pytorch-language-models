import torch
from torch import nn
import typing, time
import torch.nn.functional as F
from typing import List, Tuple



class CharacterEmbeddingLayer(nn.Module):
    '''
    A generic module to embed characters using 2d convolutions.
    Params:
     - char_vocab_size: size of char vocab of the dataset
     - char_emb_dim: size of low dimensional representation of each character.
                     Generally < char_vocab_dim
     - filters: list of tuples with the first value of tuple denoting the number
                out_channels/filters and second value denoting the width of the
                filter 
    
    This module takes in the character represention of the word of dimension
    [batch_size, seq_len, max_word_len]. max_word_len is the length of the 
    longest word in the training set. This is then passed to an embedding layer
    which returns a low-dimensional representation of each character within 
    a word. This matrix is then convolved with a number of filters of varying
    widths (as in param filters). The kernel size for each convolution is 
    [char_emb_dim, width of the current filter]. The convolved feature maps 
    are then max-pooled over time to select only one feature for a particular 
    filter. These singular features are then stacked together and result in a
    feature vector whose size depends on the total number of filters.
    '''
    
    def __init__(self, char_vocab_size:int, char_emb_dim:int, filters:List[tuple]):
        
        super().__init__()
        
        self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim)
        
        self.filters = filters
        
        self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels=1, 
                                                    out_channels=out_channels, 
                                                    kernel_size=(char_emb_dim, width), 
                                                    bias=True) for out_channels, width in filters])
        
    
    def forward(self, x):
        # x = [bs, seq_len, max_word_len]
        
        batch_size = x.shape[0]
        max_word_len = x.shape[-1]
        
        
        x = x.view(-1, max_word_len)
        # x = [bs*seq_len, max_word_len]
        
        embed = self.char_embedding(x)
        # embed = [bs*seq_len, max_word_len, char_emb_dim]
        
        embed = embed.unsqueeze(1)
        # embed = [bs*seq_len, 1, max_word_len, char_emb_dim]
        
        embed = embed.permute(0,1,3,2)
        # embed = [bs*seq_len, 1, char_emb_dim, max_word_len]
        
        feature_list = []
        
        for conv_layer in self.conv_layers:
            
            feature_map = torch.tanh(conv_layer(embed))
            # feature_map = [bs*seq_len, out_channels, 1, max_word_len - width + 1]
            
            feature = torch.max(feature_map, dim=-1)[0]
            # feature = [bs*seq_len, out_channels, 1]
            
            feature = feature.squeeze()
            # [bs*seq_len, out_channels]
            
            feature_list.append(feature)
        
        
        char_embedding = torch.cat(feature_list, dim=1)
        # [bs*seq_len, out_channels]
        
        out_channels = char_embedding.shape[-1]
        
        char_embedding = char_embedding.view(batch_size, -1, out_channels)
        # [bs, seq_len, out_channels]
        
        return char_embedding




class HighwayNetwork(nn.Module):
    '''
    A generic module for creating highway networks. This module does not
    alter the shape of input tensor. It is generally used to improve
    backpropagation and make training faster but there are other interpretations
    of using this too.
    '''
    
    def __init__(self, input_dim:int, num_layers=1):
        
        super().__init__()
        
        self.num_layers = num_layers
        
        self.flow_layer = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])
        self.gate_layer = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])
        
    def forward(self, x):
        
        for i in range(self.num_layers):
            
            flow_value = F.relu(self.flow_layer[i](x))
            gate_value = torch.sigmoid(self.gate_layer[i](x))
            
            x = gate_value * flow_value + (1-gate_value) * x
        
        return x



class CharacterAwareNeuralLM(nn.Module):
    '''
    The main class of the implemented paper. This module takes in the character-level 
    representation of the word and passes it to the CharacterEmbeddeding to get a continuous
    representation which is essentially same as using word-embeddings. This tensor is then 
    passed to the highway network and then to a 2-layer LSTM network. 
    It should be noted that although the input to this model is at character-level, the predictions
    are still made at a word-level.
    Params:
     - hidden_dim: LSTM hidden size
     - num_lstm_layers: number of LSTM layers
     - word_vocab_size: a linear layer with output_features as this value is used to get 
                        word-level predictions.
    '''
    
    def __init__(self, char_vocab_size:int, char_emb_dim:int, filters:List[tuple], hidden_dim:int, num_lstm_layers:int, word_vocab_size:int):
        
        super().__init__()
        
        self.character_embedding = CharacterEmbeddingLayer(char_vocab_size, char_emb_dim, filters)
        
        highway_input_dim = sum([x for x,y in filters])
        
        self.highway_network = HighwayNetwork(highway_input_dim)
        
        self.lstm = nn.LSTM(input_size=highway_input_dim, 
                            hidden_size=hidden_dim, 
                            batch_first=True, 
                            num_layers=num_lstm_layers,
                            dropout=0.5)
        
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(hidden_dim, word_vocab_size)
        
    def forward(self, x):
        # x = [bs, seq_len, max_word_len]
        
        char_embed = self.character_embedding(x)
        # [bs, seq_len, highway_dim]
        
        highway_out = self.highway_network(char_embed)
        # [bs, seq_len, highway_dim]
        
        lstm_out, _ = self.lstm(highway_out)
        # [bs, seq_len, hidden_dim]
        
        lstm_out = self.dropout(lstm_out)
        
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        # [bs*seq_len, hidden_dim]
        
        out = self.linear(lstm_out)
        # [bs*seq_len, word_vocab_size]
        
        return out