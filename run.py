from data import DataIterator, CharacterLanguageModelingDataset
from vocab import Vocab
from runner import Runner
from models.character_aware_nlm import CharacterEmbeddingLayer, HighwayNetwork, CharacterAwareNeuralLM
from tokenizer import Tokenizer
import torch
from torch import nn
import typing, time
import torch.nn.functional as F
from typing import List, Tuple




BATCH_SIZE = 20
SEQUENCE_LENGTH = 35
tokenizer = Tokenizer('split', lower=True, eos_token='+')
device = torch.device('cuda')
FILTERS = [(25,1), (50,2), (75,3), (100,4), (125,5), (150,6)]
HIDDEN_DIM = 300
NUM_LSTM_LAYERS = 2
CHAR_EMB_DIM = 15

dataset = CharacterLanguageModelingDataset(train_data_path='ptb-train.txt', 
                                           valid_data_path='ptb-valid.txt', 
                                           test_data_path='ptb-test.txt', 
                                           tokenizer= tokenizer)


train_iterator, valid_iterator, test_iterator = dataset.load_iterators(BATCH_SIZE, SEQUENCE_LENGTH)

model = CharacterAwareNeuralLM(char_vocab_size=len(dataset.char_vocab), 
                               char_emb_dim=CHAR_EMB_DIM, 
                               filters=FILTERS, 
                               hidden_dim=HIDDEN_DIM, 
                               num_lstm_layers=NUM_LSTM_LAYERS, 
                               word_vocab_size=len(dataset.word_vocab)).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

iterators = {'train_iterator':train_iterator, 
             'valid_iterator': valid_iterator, 
             'test_iterator':test_iterator}
             
runner = Runner(model=model, 
                optimizer=optimizer, 
                iterators=iterators, 
                device=device,
                epochs=3)

runner()




