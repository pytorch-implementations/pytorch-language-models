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
    def __init__(self, train_data_path, valid_data_path, test_data_path, tokenizer, **vocab_kwargs):

        assert isinstance(train_data_path, str)
        assert isinstance(valid_data_path, str)
        assert isinstance(test_data_path, str)
        assert isinstance(tokenizer, Tokenizer)

        self.train_data = self._load_dataset(train_data_path)
        self.valid_data = self._load_dataset(valid_data_path)
        self.test_data = self._load_dataset(test_data_path)
        self.tokenizer = tokenizer
        self.vocab = self._create_vocab(self.train_data, **vocab_kwargs)

    def _load_dataset(self, data_path: str) -> list:
        '''
        Reads dataset from path. Appends each line in a list.
        Returns a list of strings.
        '''

        data = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(line)

        return data

    def _create_vocab(self, data, **vocab_kwargs):
        '''
        Create vocabulary from training data. Uses our vocab script.
        A list of tokenized examples is passed to our script's method.
        Tokenization also uses our script.
        '''

        self.vocab = vocab.build_vocab_from_iterator([self.tokenizer(line) for line in data], **vocab_kwargs)

        return self.vocab

    def _get_tokens(self, data):
        '''
        Returns the whole dataset as a list of tokens.
        '''

        tokens = []

        for line in data:
            tokens.extend(self.tokenizer(line))

        return tokens

    def _to_ids(self, data, batch_size):
        '''
        Converts tokens/words to ids. Returns a tensor of the format
        [batch_size, *].
        '''

        tokens = self._get_tokens(data)
        token_count = 0
        ids = torch.LongTensor(len(tokens))
        for token in tokens:
            ids[token_count] = self.vocab[token]
            token_count += 1

        assert token_count == len(tokens)

        n_batches = ids.size(0) // batch_size
        ids = ids[:n_batches * batch_size]
        return ids.view(batch_size, -1)

    def _get_iterator(self, data, batch_size, sequence_length):
        '''
        Returns a list iterator object by getting the data in LM format.
        Targets are calculated by shifting the input sequence by 1 time-step.
        '''

        ids = self._to_ids(data, batch_size)
        iterator = []
        for i in range(0, ids.size(1) - sequence_length, sequence_length):
            inputs = ids[:, i:i+sequence_length]
            targets = ids[:, (i+1):(i+1)+sequence_length]
            iterator.append((inputs, targets))

        return iter(iterator)

    def load_iterators(self, batch_size, sequence_length):

        assert isinstance(batch_size, int)
        assert isinstance(sequence_length, int)

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
