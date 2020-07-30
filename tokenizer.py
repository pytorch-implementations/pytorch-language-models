from typing import Callable, List


class Tokenizer:
    '''
    Class that handles all of the tokenization (string to list of strings)
    Actual tokenization is done by the `tokenize_fn`, can either be a string
    which uses one of the provided tokenizers from `get_tokenizer` or a callable
    function if the user wants to specify their own.

    lower if true lowercase each token AFTER tokenization
    sos_token is a token appended to the start of each sequence
    eos_token is a token appended to the end of each sequence
    max_length is the maximum length of a sequence
      any sequences longer than max_length are trimmed (from the end)
      if sos/eos tokens are not None, then they count towards the maximum length
      however the sequences are trimmed accordingly before they are added and
      the sos/eos tokens will always be included in the sequence
    '''

    def __init__(self, tokenize_fn, lower: bool = False, sos_token: str = None, eos_token: str = None, max_length: int = None):

        assert callable(tokenize_fn) or isinstance(tokenize_fn, str), f"tokenize_fn should be callable or str, got {type(tokenize_fn)}"
        assert isinstance(lower, bool), f"lower should be bool, got {type(lower)}"
        assert isinstance(sos_token, str) or sos_token is None, f"sos_token should be str or None, got {type(sos_token)}"
        assert isinstance(eos_token, str) or eos_token is None, f"sos_token should be str or None, got {type(sos_token)}"
        assert (isinstance(max_length, int) and max_length > 0) or max_length is None, f"max_length should be a positive integer or None, got {max_length} ({type(max_length)})"

        self.tokenize_fn = tokenize_fn if callable(tokenize_fn) else get_tokenizer(tokenize_fn)
        self.lower = lower
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.max_length = max_length

    def tokenize(self, example: str) -> List[str]:

        assert isinstance(example, str), f"example should be a str, got {type(example)}"

        tokens = self.tokenize_fn(example)

        assert isinstance(tokens, list), f"output of tokenize_fn should be a list, got {type(tokens)}"
        assert len(tokens) > 0, f"got empty list of tokens from tokenizing input: {example}"
        assert all([isinstance(token, str) for token in tokens]), f"output of tokenize_fn should be a list of strings, got {[type(token) for token in tokens]}"

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

    def __call__(self, example: str) -> List[str]:
        return self.tokenize(example)


def get_tokenizer(tokenizer: str) -> Callable:
    '''
    Gets one of the provided tokenization functions, a function which takes
    in a string and returns a list of strings
    '''

    assert isinstance(tokenizer, str), f"input to get_tokenizer should be a str, got {type(tokenizer)}"

    if tokenizer == "split":
        def _split_tokenize(example):
            return example.split()
        return _split_tokenize

    if tokenizer == "nltk":
        import nltk
        nltk.download("punkt")
        tokenizer = nltk.tokenize.word_tokenize
        return tokenizer

    elif tokenizer == "punkt":
        import nltk
        tokenizer = nltk.tokenize.wordpunct_tokenize
        return tokenizer

    elif tokenizer == "tweet":
        import nltk
        tokenizer = nltk.tokenize.casual.TweetTokenizer()
        return tokenizer.tokenize

    elif tokenizer == "ptb":
        import nltk
        tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
        return tokenizer.tokenize

    elif tokenizer == "spacy":
        import spacy
        from functools import partial
        spacy = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])
        def _spacy_tokenize(example, spacy):
            return [token.text for token in spacy.tokenizer(example)]

        return partial(_spacy_tokenize, spacy=spacy)

    else:
        raise ValueError(f'{tokenizer} is not a recognized tokenizer.')

if __name__ == "__main__":
    example = "HELLO world HOW are YOU TODAY? i am feeling very good. hope you're happy to hear! because i am."
    expected_tokens = ["<sos>", "hello", "world", "how", "<eos>"]
    tokenize_fn = lambda x : x.split()
    tokenizer = Tokenizer(tokenize_fn, lower=True, sos_token="<sos>", eos_token="<eos>", max_length=5)
    tokens = tokenizer(example)
    assert tokens == expected_tokens

    import time
    for tokenizer_fn in ["split", "nltk", "punkt", "tweet", "ptb", "spacy"]:
        print(tokenizer_fn)
        tokenizer = Tokenizer(tokenizer_fn)
        t0 = time.monotonic()
        for _ in range(100_000):
            tokens = tokenizer(example)
        dt = time.monotonic() - t0
        print("\t", tokens)
        print("\t", dt)
