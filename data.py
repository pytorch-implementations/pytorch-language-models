import tokenizer
import vocab
import torch
from typing import List, Tuple, Iterator
from tokenizer import Tokenizer


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

        assert isinstance(data_path, str), f"data_path should be a str, got {type(data_path)}"

        data = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.extend(self.tokenizer(line))

        return data

    def _create_vocab(self, data: List[str], **vocab_kwargs) -> vocab.Vocab:
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
