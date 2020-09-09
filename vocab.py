from typing import Counter, Dict, List, Tuple
import collections
import torch


class Vocab:
    """
    Class to handle a vocabulary, a mapping between strings and a their corresponding integer values.
    Vocabulary must be created with a counter where each key is a token and each value is the number
    of times that tokens appears in the training dataset.
    """

    def __init__(self, counter: Counter, min_freq: int = 1, max_size: int = None, unk_token: str = "<unk>", pad_token: str = "<pad>", special_tokens: List[str] = None):

        assert isinstance(counter, collections.Counter), f"counter should be a collections.Counter, got {type(counter)}"
        assert isinstance(min_freq, int) and min_freq > 0, f"min_freq should an integer greater than 0, got {min_freq} ({type(min_freq)})"
        assert (isinstance(max_size, int) and max_size > 0) or max_size is None, f"max_size should be an integer greater than 0 or None, got {max_size} ({type(max_size)})"
        assert isinstance(unk_token, str) or unk_token is None, f"unk_token should be a string or None, got {unk_token} ({type(unk_token)})"
        assert isinstance(pad_token, str) or pad_token is None, f"pad_token should be a string or None, got {pad_token} ({type(unk_token)})"
        assert (isinstance(special_tokens, list) and all([isinstance(token, str) for token in special_tokens])) or special_tokens is None, f"special_tokens should be a list of strings, got {special_tokens}"

        self.counter = counter
        self.min_freq = min_freq
        self.max_size = max_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.specials = special_tokens

        self._stoi, self._itos = self._create_vocab(counter, min_freq, max_size, unk_token, pad_token, special_tokens)

    def _create_vocab(self, counter: Counter, min_freq: int, max_size: int, unk_token: str, pad_token: str, special_tokens: List[str]) -> Tuple[Dict, List]:
        """
        Does the actual vocabulary creation
        - tokens that appear less than min_freq times are ignored
        - once the vocabulary reaches max size, no more tokens are added
        - unk_token is the token used to replace tokens not in the vocabulary
        - pad_token is used to pad sequences
        - special tokens are other tokens we want appended to the start of our vocabulary
        """

        stoi = dict()

        if pad_token is not None:
            stoi[pad_token] = len(stoi)
        if unk_token is not None:
            stoi[unk_token] = len(stoi)
        if special_tokens is not None:
            for special in special_tokens:
                stoi[special] = len(stoi)

        for token, count in counter.most_common(max_size):
            if count >= min_freq:
                stoi[token] = len(stoi)
            else:
                break

        itos = [token for token, index in stoi.items()]

        assert len(stoi) > 0, "Created vocabulary is empty!"
        assert max_size is None or len(stoi) <= max_size, "Created vocabulary is larger than max size"
        assert len(stoi) == len(itos), "Created str -> int vocab length is not the same size as the int -> str vocab length"

        return stoi, itos

    def stoi(self, token: str) -> int:
        """
        Converts a token (str) into its corresponding integer value from the vocabulary
        If the token is not in the vocabulary, returns the integer value of the unk_token
        If unk_token is set to None, throws an error
        """

        assert isinstance(token, str), f"Input to vocab.stoi should be str, got {type(token)}"

        if token in self._stoi:
            return self._stoi[token]
        else:
            assert self.unk_token is not None
            return self._stoi[self.unk_token]

    def itos(self, index: int) -> str:
        """
        Converts an integer into its corresponding token (str) from the vocabulary
        If the integer value is outside of the vocabulary range, throws an error
        """

        assert isinstance(index, int), f"Input to vocab.itos should be an integer, got {type(index)}"
        assert index >= 0, f"Input to vocab.itos should be a positive integer (or zero), got {index}"
        assert index < len(self._itos), f"Input index out of range, should be <{len(self._itos)}, got {index}"

        return self._itos[index]

    def save(self, save_path):
        """
        Saves the vocabulary object to the given save_path
        """
        
        assert isinstance(save_path, str), f"save_path should be a str, got {type(save_path)}"
        
        torch.save(self, save_path)

    def __getitem__(self, x):
        """
        Convenience function so we can call vocab[x] and if x is a string then
        vocab.stoi(x) is called, and if x is an integer then vocab.itos(x) is called
        """

        if isinstance(x, str):
            return self.stoi(x)
        elif isinstance(x, int):
            return self.itos(x)
        else:
            raise ValueError(f'When calling vocab[x], x should be either an int or str, got {type(x)}')


def build_vocab_from_iterator(iterator, **kwargs):

    assert isinstance(iterator, list), f"iterator should be a list, got {type(examples)}"
    assert all([isinstance(i, (list, str)) for i in iterator]), "iterator should be a list of lists or strings"

    counter = collections.Counter()

    for i in iterator:
        if isinstance(i, list):
            counter.update(i)
        else:
            counter[i] += 1

    vocab = Vocab(counter, **kwargs)

    return vocab


if __name__ == "__main__":
    examples = [["hello", "world", "hello"], ["hello", "magic", "world"]]

    vocab = build_vocab_from_iterator(examples)
    expected_vocab_itos = ["<unk>", "<pad>", "hello", "world", "magic"]
    expected_vocab_stoi = {"<unk>": 0, "<pad>": 1, "hello": 2, "world": 3, "magic": 4}

    assert len(vocab._itos) == 5
    assert expected_vocab_itos == vocab._itos
    assert expected_vocab_stoi == vocab._stoi
    assert vocab.stoi("zebra") == vocab.stoi(vocab.unk_token)
    assert vocab.itos(1) == vocab.itos(vocab.stoi(vocab.pad_token))
    assert vocab["magic"] == vocab.stoi("magic") == 4
    assert vocab[3] == vocab.itos(3) == "world"

    vocab.save('test_vocab.pt')
    vocab = torch.load('test_vocab.pt')

    assert len(vocab._itos) == 5
    assert expected_vocab_itos == vocab._itos
    assert expected_vocab_stoi == vocab._stoi
    assert vocab.stoi("zebra") == vocab.stoi(vocab.unk_token)
    assert vocab.itos(1) == vocab.itos(vocab.stoi(vocab.pad_token))
    assert vocab["magic"] == vocab.stoi("magic") == 4
    assert vocab[3] == vocab.itos(3) == "world"

    vocab = build_vocab_from_iterator(examples, min_freq=2)
    expected_vocab_itos = ["<unk>", "<pad>", "hello", "world"]
    expected_vocab_stoi = {"<unk>": 0, "<pad>": 1, "hello": 2, "world": 3}

    assert len(vocab._itos) == 4
    assert expected_vocab_itos == vocab._itos
    assert expected_vocab_stoi == vocab._stoi
    assert vocab.stoi("zebra") == vocab.stoi(vocab.unk_token)
    assert vocab.stoi("magic") == vocab.stoi(vocab.unk_token)
    assert vocab.itos(1) == vocab.itos(vocab.stoi(vocab.pad_token))
    assert vocab["hello"] == vocab.stoi("hello") == 2
    assert vocab[3] == vocab.itos(3) == "world"

    vocab.save('test_vocab.pt')
    vocab = torch.load('test_vocab.pt')

    assert len(vocab._itos) == 4
    assert expected_vocab_itos == vocab._itos
    assert expected_vocab_stoi == vocab._stoi
    assert vocab.stoi("zebra") == vocab.stoi(vocab.unk_token)
    assert vocab.stoi("magic") == vocab.stoi(vocab.unk_token)
    assert vocab.itos(1) == vocab.itos(vocab.stoi(vocab.pad_token))
    assert vocab["hello"] == vocab.stoi("hello") == 2
    assert vocab[3] == vocab.itos(3) == "world"

    import os
    os.remove('test_vocab.pt')
