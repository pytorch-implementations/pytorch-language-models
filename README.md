# pytorch-language-models

Proposed pipeline:
``` python
# our modules
import tokenizer
import data
import vocab
import models
import trainer

# use existing tokenizers from string or define your own callable function
tokenizer = tokenizer.Tokenizer('nltk')

# load data
train_data = data.load_data(train_path, tokenizer)
valid_data = data.load_data(valid_path, tokenizer)
test_data = data.load_data(test_path, tokenizer)

# create vocabulary
vocab = vocab.load_from_iterator(train_data)

# create data iterators
train_iterator = data.load_iterator(train_data, vocab)
valid_iterator = data.load_iterator(valid_data, vocab)
test_iterator = data.load_iterator(test_data, vocab)

# get model details from config file
config = config.load_config(config_path)

# load model from vocab and config
model = models.load_model(vocab, config)

# create trainer
trainer = Trainer(model)

# run training
while trainer.run:

    # train model on training data
    trainer.train(train_iterator)
    # evaluate model on validation data
    trainer.eval(valid_iterator)

# test model on test data
trainer.eval(test_iterator)

# generate some text
model.generate()

# save tokenizer
tokenizer.save(tokenizer_path)
# save vocab
vocab.save(vocab_path)
# save model
model.save(model_path)
```