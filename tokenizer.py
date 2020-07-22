class Tokenizer:
    def __init__(self, tokenize_fn, lower=False, sos_token=None, eos_token=None, max_length=None):

        assert callable(tokenize_fn)
        assert isinstance(lower, bool)
        assert isinstance(sos_token, str) or sos_token is None
        assert isinstance(eos_token, str) or eos_token is None
        assert (isinstance(max_length, int) and max_length > 0) or max_length is None

        self.tokenize_fn = tokenize_fn
        self.lower = lower
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.max_length = max_length

    def tokenize(self, example):

        assert isinstance(example, str)

        tokens = self.tokenize_fn(example)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all([isinstance(token, str) for token in tokens])

        if self.lower:
            tokens = [token.lower() for token in tokens]

        if self.max_length is not None:
            length = self.max_length
            if self.sos_token is not None:
                length -= 1
            if self.eos_token is not None:
                length -= 1
            tokens = tokens[:length]

        if self.sos_token is not None:
            tokens = [self.sos_token] + tokens

        if self.eos_token is not None:
            tokens = tokens + [self.eos_token]

        return tokens

    def __call__(self, example):
        return self.tokenize(example)


if __name__ == "__main__":
    example = "HELLO world HOW are YOU TODAY?"
    expected_tokens = ['<sos>', 'hello', 'world', 'how', '<eos>']
    tokenize_fn = lambda x : x.split()
    tokenizer = Tokenizer(tokenize_fn, lower=True, sos_token='<sos>', eos_token='<eos>', max_length=5)
    tokens = tokenizer(example)
    assert tokens == expected_tokens
