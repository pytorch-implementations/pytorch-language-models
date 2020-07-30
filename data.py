import tokenizer
import vocab
import torch
from tokenizer import Tokenizer



## For PTB dataset. 
## Link for data: https://github.com/wojzaremba/lstm/tree/master/data
## Idea for preprocessing:
## https://github.com/yunjey/pytorch-tutorial/blob/0500d3df5a2a8080ccfccbc00aca0eacc21818db/tutorials/02- intermediate/language_model/data_utils.py#L25

## https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/language_model/main.py#L30-L50

## TODO
## - Review and make sure if this is the way to preprocess data for LM tasks.
## - Ask Ben for a review.
## - Add more sanity checks
## - Test for other datasets

## Changes pipeline in the main.py a bit. 


class LanguageModelingDataset:
    """
    Takes in dataset paths and returns data iterators for each dataset.
    Currently works for PTB dataset. Will have to check for other datasets.
    """
    def __init__(self, train_data_path, valid_data_path, test_data_path, tokenizer, batch_size, sequence_length, **kwargs):

        assert isinstance(train_data_path, str)
        assert isinstance(valid_data_path, str)
        assert isinstance(test_data_path, str)
        assert isinstance(tokenizer, Tokenizer)
        assert isinstance(batch_size, int)
        assert isinstance(sequence_length, int)

        self.train_data = self.load_dataset(train_data_path)
        self.valid_data = self.load_dataset(valid_data_path)
        self.test_data = self.load_dataset(test_data_path)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab = self.create_vocab(**kwargs)

    def load_dataset(self, data_path: str) -> list:
        '''
        Reads dataset from path. Appends each line in a list.
        Returns a list of strings.
        '''

        data = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(line)

        return data

    def create_vocab(self, **kwargs):
        '''
        Create vocabulary from training data. Uses our vocab script.
        A list of tokenized examples is passed to our script's method.
        Tokenization also uses our script.
        '''

        tokenized_data = []
        for line in self.train_data:
            tokens = self.tokenizer(line)
            tokenized_data.append(tokens)

        self.vocab = vocab.build_vocab_from_iterator(tokenized_data, **kwargs)

        return self.vocab

    def get_tokens(self, data):
        '''
        Returns the whole dataset as a list of tokens.
        '''

        tokens = []

        for line in data:
            tokens.extend(self.tokenizer(line))

        return tokens

    def to_ids(self, data):
        '''
        Converts tokens/words to ids. Returns a tensor of the format
        [batch_size, *].
        '''

        tokens = self.get_tokens(data)
        token_count = 0
        ids = torch.LongTensor(len(tokens))
        for token in tokens:
            ids[token_count] = self.vocab[token]
            token_count += 1

        assert token_count == len(tokens)

        num_batches = ids.size(0) // self.batch_size
        ids = ids[:num_batches * self.batch_size]
        return ids.view(self.batch_size, -1)

    def get_iterator(self, data):
        '''
        Returns a list iterator object by getting the data in LM format.
        Targets are calculated by shifting the input sequence by 1 time-step.
        '''

        ids = self.to_ids(data)
        iterator = []
        for i in range(0, ids.size(1) - self.sequence_length, self.sequence_length):
            inputs = ids[:, i:i+self.sequence_length]
            targets = ids[:, (i+1):(i+1)+self.sequence_length]
            iterator.append((inputs, targets))

        return iter(iterator)

    def load_iterators(self):

        train_iterator = self.get_iterator(self.train_data)
        valid_iterator = self.get_iterator(self.valid_data)
        test_iterator = self.get_iterator(self.test_data)

        return train_iterator, valid_iterator, test_iterator


if __name__ == "__main__":

    tokenizer = tokenizer.Tokenizer('split', lower=True, eos_token='<eos>')
    data = LanguageModelingDataset('ptb-train.txt',
                                   'ptb-valid.txt',
                                   'ptb-valid.txt',
                                   tokenizer,
                                   20,
                                   30)

    train_iterator, valid_iterator, test_iterator = data.load_iterators()
    x = next(iter(train_iterator))
    assert(type(x)) == tuple
    assert(len(x)) == 2
    # x[0] = input, x[1] = target
    assert x[0].shape == torch.Size([20, 30])
    assert x[1].shape == torch.Size([20, 30])

    # example
    print("Sample Input: \n", x[0][0])
    print("Sample Output: \n", x[1][0])
