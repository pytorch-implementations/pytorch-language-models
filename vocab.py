import collections


class Vocab:
    def __init__(self, counter, min_freq=1, max_size=None, unk_token="<unk>", pad_token="<pad>", special_tokens=None):

        assert isinstance(counter, collections.Counter)
        assert isinstance(min_freq, int) and min_freq > 0
        assert (isinstance(max_size, int) and max_size > 0) or max_size is None
        assert isinstance(unk_token, str) or unk_token is None
        assert isinstance(pad_token, str) or pad_token is None
        assert (isinstance(special_tokens, list) and all([isinstance(token, str) for token in special_tokens])) or special_tokens is None

        self.counter = counter
        self.min_freq = min_freq
        self.max_size = max_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.specials = special_tokens

        self._stoi, self._itos = self._create_vocab(counter, min_freq, max_size, unk_token, pad_token, special_tokens)

    def _create_vocab(self, counter, min_freq, max_size, unk_token, pad_token, special_tokens):

        stoi = dict()

        if unk_token is not None:
            stoi[unk_token] = len(stoi)
        if pad_token is not None:
            stoi[pad_token] = len(stoi)
        if special_tokens is not None:
            for special in special_tokens:
                stoi[special] = len(stoi)

        for token, count in counter.most_common(max_size):
            if count >= min_freq:
                stoi[token] = len(stoi)
            else:
                break

        itos = [token for token, index in stoi.items()]

        assert len(stoi) > 0
        assert max_size is None or len(stoi) <= max_size
        assert len(stoi) == len(itos)

        return stoi, itos

    def stoi(self, token):

        assert isinstance(token, str)

        if token in self._stoi:
            return self._stoi[token]
        else:
            assert self.unk_token is not None
            return self._stoi[self.unk_token]

    def itos(self, index):

        assert isinstance(index, int)
        assert index >= 0
        assert index < len(self._itos)

        return self._itos[index]

    def __getitem__(self, x):
        if isinstance(x, str):
            return self.stoi(x)
        elif isinstance(x, int):
            return self.itos(x)
        else:
            raise ValueError


def build_vocab_from_iterator(examples, **kwargs):

    assert isinstance(examples, list)
    assert all([isinstance(example, list) for example in examples])

    counter = collections.Counter()

    for example in examples:
        assert all([isinstance(token, str) for token in example])
        counter.update(example)

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
