# pytorch-language-models

Proposed pipeline:
``` python
# using nltk for tokenization, can use anything!
import nltk
nltk.download('punkt')  # load tokenizer
tokenize_fn = nltk.tokenize.word_tokenize  # define tokenizer function

# our modules
import tokenizer
import data
import vocab
import models
import trainer

tokenizer = tokenizer.Tokenizer(tokenize_fn)  # create tokenizer

train_data = data.load_data(train_path, tokenizer)  # load data
valid_data = data.load_data(valid_path, tokenizer)
test_data = data.load_data(test_path, tokenizer)

vocab = vocab.load_from_iterator(train_data)  # create vocabulary

train_iterator = data.load_iterator(train_data, vocab)  # create data iterators
valid_iterator = data.load_iterator(valid_data, vocab)
test_iterator = data.load_iterator(test_data, vocab)

config = config.load_config(config_path)  # get model details from config file

model = models.load_model(vocab, config)  # load model from vocab and config

trainer = Trainer(model)  # create trainer

while trainer.run:  # run training

    trainer.train(train_iterator)  # train model on training data
    trainer.eval(valid_iterator)  # evaluate model on validation data

trainer.eval(test_iterator)  # test model on test data

model.generate()  # generate some text

tokenizer.save(tokenizer_path)  # save tokenizer
vocab.save(vocab_path)  # save vocab
model.save(model_path)  # save model
```