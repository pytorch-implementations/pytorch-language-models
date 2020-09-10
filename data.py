from typing import List, Tuple, Iterator
from tokenizer import Tokenizer
import vocab


class WordLanguageModelingDataset:
    """
    Class to word level datasets. Given train/valid/test paths, a tokenizer and vocab arguments it will:
      - tokenize each line of the given data paths and extend to a list
      - create the vocabulary from the tokenized train data
    """
    def __init__(self, train_data_path: str, valid_data_path: str, test_data_path: str, tokenizer: Tokenizer, **vocab_kwargs):

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

    def _create_vocab(self, data: List[str], **vocab_kwargs) -> vocab.Vocab:
        """
        Create vocabulary from data using the Vocab class with given arguments.
        Expects the data to already be tokenized, usually from _load_dataset.
        Returns a Vocab.
        """

        assert isinstance(data, list), f"data should be a list, got {type(data)}"

        self.vocab = vocab.build_vocab_from_iterator(data, **vocab_kwargs)

        return self.vocab


class CharacterLanguageModelingDataset(WordLanguageModelingDataset):
    def __init__(self, dataset: WordLanguageModelingDataset, tokenizer: Tokenizer, **vocab_kwargs):

        self.tokenizer = tokenizer
        self.train_data = self._load_character_dataset(dataset.train_data)
        self.valid_data = self._load_character_dataset(dataset.valid_data)
        self.test_data = self._load_character_dataset(dataset.test_data)
        self.vocab = self._create_vocab(dataset.train_data, **vocab_kwargs)

    def _load_character_dataset(self, data: List[str]) -> List[List[str]]:

        data = [self.tokenizer(datum) for datum in data]

        return data


if __name__ == "__main__":

    word_tokenizer = Tokenizer('split', lower=False)
    word_dataset = WordLanguageModelingDataset('ptb.train.txt',
                                               'ptb.valid.txt',
                                               'ptb.valid.txt',
                                               word_tokenizer)
    char_tokenizer = Tokenizer('list', lower=True, sos_token='<sow>', eos_token='<eow>')
    char_dataset = CharacterLanguageModelingDataset(word_dataset,
                                                    char_tokenizer)

    print(word_dataset[:10])
    print(char_dataset[:10])
