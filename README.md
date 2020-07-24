# PyTorch Language Models

TODO:

- [] tokenizer save
- [] vocab save
- [] data load datasets
- [] data load iterators
- [] load model config
- [] load runner config
- [] load model
- [] save model
- [] basic lstm model
- [] runner class
- [] runner train
- [] runner eval
- [] runner save




Current pipeline
```
tokenizer = tokenizer.Tokenizer(args.tokenize_fn,
                                args.lower,
                                args.sos_token,
                                args.eos_token,
                                args.max_length)

# load data
train_data, valid_data, test_data = data.load_datasets(args.train_data_path,
                                                       args.valid_data_path,
                                                       args.test_data_path)

# create vocabulary
vocab = vocab.load_from_iterator(train_data,
                                 args.min_freq,
                                 args.max_size,
                                 args.unk_token,
                                 args.pad_token,
                                 args.special_tokens)

# create data iterators
train_iterator, valid_iterator, test_iterator = data.load_iterators(train_data,
                                                                    valid_data,
                                                                    test_data,
                                                                    vocab,
                                                                    args.batch_size,
                                                                    args.sequence_length)

```


After data.py additions-
```
tokenizer = tokenizer.Tokenizer(args.tokenize_fn,
                                args.lower,
                                args.sos_token,
                                args.eos_token,
                                args.max_length)

data = data.LanguageModelingDataset(args.train_data_path,
                                    args.valid_data_path,
                                    args.test_data_path,
                                    tokenizer,
                                    batch_size,
                                    sequence_length)
                                    
train_iterator, valid_iterator, test_iterator = data.load_iterators()


```
