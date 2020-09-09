from typing import List, Tuple, Iterator
import numpy as np
import pandas as pd
import torch
from collections import Counter
from tokenizer import get_tokenizer, Tokenizer
from vocab import Vocab, build_vocab_from_iterator

class LanguageModelingDataset:
    """
    Class to handle datasets. Given train/valid/test paths, a tokenizer and vocab arguments it will:
      - tokenize each line of the given data paths and extend to a list
      - create the vocabulary from the tokenized train data
    """
    def __init__(self, train_data_path, valid_data_path, test_data_path, tokenizer, **vocab_kwargs):

        assert isinstance(train_data_path, str), f"train_data_path should be a str, got {type(train_data_path)}"
        assert isinstance(valid_data_path, str), f"train_data_path should be a str, got {type(valid_data_path)}"
        assert isinstance(test_data_path, str), f"train_data_path should be a str, got {type(test_data_path)}"
        assert isinstance(tokenizer, Tokenizer), f"tokenizer should be a Tokenizer, got {type(tokenizer)}"

        self.tokenizer = tokenizer
        self.train_data = self._load_dataset(train_data_path)
        self.valid_data = self._load_dataset(valid_data_path)
        self.test_data = self._load_dataset(test_data_path)
        self.vocab = self._create_vocab(self.train_data, **vocab_kwargs)

    def _load_dataset(self, data_path: str) -> List[str]:
        """
        Reads dataset from path. Tokenizes each line and extends a list.
        """
        print("Loading data...")
        assert isinstance(data_path, str), f"data_path should be a str, got {type(data_path)}"

        data = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.extend(self.tokenizer(line))

        return data

    def _create_vocab(self, data: List[str], **vocab_kwargs) -> Vocab:
        """
        Create vocabulary from data using the Vocab class with given arguments.
        Expects the data to already be tokenized, usually from _load_dataset.
        Returns a Vocab.
        """

        assert isinstance(data, list), f"data should be a list, got {type(data)}"

        self.vocab = vocab.build_vocab_from_iterator(data, **vocab_kwargs)

        return self.vocab

    def load_iterators(self, batch_size: int, sequence_length: int) -> Tuple[Iterator, Iterator, Iterator]:
        """
        Gets the train/valid/test iterators.
        """

        assert isinstance(batch_size, int), f"batch_size should be an int, got {type(batch_size)}"
        assert isinstance(sequence_length, int), f"sequence_length should be an int, got {type(sequence_length)}"
        assert batch_size > 0, f"batch_size should be >0, got {batch_size}"
        assert sequence_length > 0, f"sequence_length should be >0, got {sequence_length}"

        train_iterator = self._get_iterator(self.train_data, batch_size, sequence_length)
        valid_iterator = self._get_iterator(self.valid_data, batch_size, sequence_length)
        test_iterator = self._get_iterator(self.test_data, batch_size, sequence_length)

        return train_iterator, valid_iterator, test_iterator

    def _get_iterator(self, data: List[str], batch_size: int, sequence_length: int) -> Iterator[Tuple[torch.LongTensor, torch.LongTensor]]:
        """
        Given some data (list of tokens), converts to ids and batches then creates a list iterator.
        Targets are calculated by shifting the input sequence by 1 time-step.
        """

        ids = self._to_ids(data, batch_size)
        iterator = []
        for i in range(0, ids.size(1) - sequence_length, sequence_length):
            inputs = ids[:, i:i+sequence_length]
            targets = ids[:, (i+1):(i+1)+sequence_length]
            iterator.append((inputs, targets))

        return iter(iterator)

    def _to_ids(self, data: List[str], batch_size: int) -> torch.LongTensor:
        """
        Converts data (the list of tokens) into an ids (integers) tensor using the vocabulary.
        Then calculates how many batches it can make from the dataset and reshapes [batch_size, *].
        """

        ids = torch.LongTensor(len(data))
        for i, token in enumerate(data):
            ids[i] = self.vocab[token]

        n_batches = ids.size(0) // batch_size
        ids = ids[:n_batches * batch_size]
        ids = ids.view(batch_size, -1)

        return ids

class DataIterator:
    '''
    Helper class to iterate through training examples for multiple epochs.
    '''
    
    def __init__(self, x, y):
        
        self.x = x
        self.y = y
    
    def __len__(self):
        return self.x.shape[0]
    
    def __iter__(self):
        
        for i in range(self.x.shape[0]):
            yield {'inputs':self.x[i], 'targets':self.y[i]}


class CharacterLanguageModelingDataset:
    '''
    A class to handle LM datasets for models that deal with character-level
    representation. Performs the following functions:
    1. Creates vocabularies for both word and character-level representation from the training
       dataset.
    2. Numericalizes the words and characters in the data.
    3. Creates iterators of the all the dataset files passed.
    
    Some important details (might be specific to https://arxiv.org/abs/1508.06615). 
    --> In order to deal with these datasets at character-level, we first replace all the
       <unk> tokens with '|' so that it can be represented by a single character. Since it
       is eventually replaced by a single id, it does not affect the word-vocab too.
    --> Additionally, we also need special tokens to mark "START OF THE WORD" and 
        "END OF THE WORD" for each word. These tokens are '{' and '}' respectively. 
        The paper claims that doing so helps the mdoel in gaining a better understanding of 
        the prefix and suffix for each word.
    ---> For <eos> token at the end of each line, we use '+' instead.
     '''
    
    def __init__(self, train_data_path:str, valid_data_path:str, test_data_path:str, tokenizer:Tokenizer):
        
        
        self.tokenizer = tokenizer

        self.train_data = self._load_dataset(train_data_path)
        self.valid_data = self._load_dataset(valid_data_path)
        self.test_data = self._load_dataset(test_data_path)
        self.word_vocab = self._prepare_word_vocab(self.train_data)
        self.char_vocab = self._prepare_char_vocab(self.train_data)
        
        
    def _load_dataset(self, data_path: str) -> dict:
        '''
        Reads dataset from path. Appends each line in a list.
        Returns a list of strings.
        '''
        print("Loading data...")
        lines = []
        word_tokens = []
        char_tokens = []
        data = {}
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.replace('<unk>', '|')
                words = self.tokenizer(line)
                word_tokens.extend(words)
                for word in words:
                    char_tokens.extend([ch for ch in word])
                lines.append(line)
        data = {'lines':lines, 'words':word_tokens, 'chars': char_tokens}
        return data
    
    
    def _prepare_word_vocab(self, data:dict) -> Vocab:
        '''
        Create a word vocab for our dataset. Uses our vocab.py script.
        '''
       
        words = data['words']
        word_counter = Counter(words)
        word_vocab = Vocab(word_counter, unk_token=None)
        return word_vocab
    
    def _prepare_char_vocab(self, data:dict) -> Vocab:
        '''
        Create a char vocab for our dataset. Uses our vocab.py script.
        '''
        
        chars = data['chars']
        char_counter = Counter(chars)
        char_vocab = Vocab(char_counter, unk_token=None, special_tokens=['{', '}'])
        return char_vocab
    
    def _to_ids(self, data) -> Tuple[np.ndarray]:
        '''
        Converts the chars/words to unique id/number.
        '''
        
        word_ids = []
        char_ids = []
        for line in data['lines']:
            for word in self.tokenizer(line):
                
                word_ids.append(self.word_vocab[word])
                char_array = [self.char_vocab[char] for char in '{' + word + '}']
                char_ids.append(char_array)
            
        
        self.max_word_len = max([len(char_list) for char_list in char_ids])
        _word_ids = np.array(word_ids, dtype=np.long)
        
        # create an empty array to pad the words that have lesser characters than
        # max_word_len
        _char_ids = np.zeros([len(char_ids), self.max_word_len],dtype=np.long)
        for i, char_id in enumerate(char_ids):
            _char_ids[i,:len(char_id)] = char_id
            
        return _word_ids, _char_ids
        
    def _get_iterator(self, data, batch_size, sequence_length) -> DataIterator:
        
        word_ids, char_ids = self._to_ids(data)
        
        # ensures that inputs and targets are created in a way to maintain the correspondence 
        # between the character inputs and word targets
        num_words = len(data['words'])
        num_complete_words = (num_words // (batch_size * sequence_length)) * (batch_size * sequence_length)
        char_ids = char_ids[:num_complete_words, :]
        word_ids = word_ids[:num_complete_words]
        
        targets = np.zeros_like(word_ids)
        
        # create a new array which shifts the target by one time-step
        targets[:-1] = word_ids[1:]
        targets[-1] = word_ids[0]
        
        
        inputs = torch.tensor(char_ids, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)
        
        inputs = inputs.view(-1, batch_size, sequence_length, self.max_word_len)
        targets = targets.view(-1, batch_size, sequence_length)
        
        iterator = DataIterator(inputs, targets)
        
        return iterator

        
    def load_iterators(self, batch_size, sequence_length) -> Tuple[DataIterator]:
        
        print("Loading iterators...")
        train_iterator = self._get_iterator(self.train_data, batch_size, sequence_length)
        valid_iterator = self._get_iterator(self.valid_data, batch_size, sequence_length)
        test_iterator = self._get_iterator(self.test_data, batch_size, sequence_length)

        return train_iterator, valid_iterator, test_iterator
    
        


if __name__ == "__main__":

    tokenizer = tokenizer.Tokenizer('split', lower=True, eos_token='<eos>')
    dataset = LanguageModelingDataset('ptb.train.txt',
                                      'ptb.valid.txt',
                                      'ptb.valid.txt',
                                       tokenizer)

    batch_size = 10
    sequence_length = 20

    train_iterator, valid_iterator, test_iterator = dataset.load_iterators(batch_size, sequence_length)
    batch = next(iter(train_iterator))
    assert(type(batch)) == tuple
    assert(len(batch)) == 2
    input, target = batch
    # x[0] = input, x[1] = target
    assert input.shape == torch.Size([batch_size, sequence_length])
    assert target.shape == torch.Size([batch_size, sequence_length])

    assert torch.all(torch.eq(input[:, 1:], target[:, :-1]))

    # example
    print("Sample Input:")
    print("\t", input[0])
    print("\t", [dataset.vocab[x.item()] for x in input[0]])
    print("Sample Output:")
    print("\t", target[0])
    print("\t", [dataset.vocab[x.item()] for x in target[0]])
