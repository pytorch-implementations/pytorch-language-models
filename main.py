import argparse

#our modules
import tokenizer
import data
import vocab
import config
import models
import runner

parser = argparse.ArgumentParser(description="PyTorch Language Models")
parser.add_argument("--tokenize_fn", type=str, default="punkt")
parser.add_argument("--lower", action="store_true")
parser.add_argument("--sos_token", type=str, default=None)
parser.add_argument("--eos_token", type=str, default=None)
parser.add_argument("--max_length", type=int, default=None)
parser.add_argument("--train_data_path", type=str, required=True)
parser.add_argument("--valid_data_path", type=str, required=True)
parser.add_argument("--test_data_path", type=str, required=True)
parser.add_argument("--min_freq", type=int, default=1)
parser.add_argument("--max_size", type=int, default=None)
parser.add_argument("--unk_token", type=str, default="<unk>")
parser.add_argument("--pad_token", type=str, default="<pad>")
parser.add_argument("--special_tokens", nargs="*", default=None)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--sequence_length", type=int, default=100)
parser.add_argument("--model_config_path", type=str, default="configs/default_model.json")
parser.add_argument("--runner_config_path", type=str, default="configs/default_runner.json")
args = parser.parse_args()

# load tokenizer
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

# load model config
model_config = config.load_model_config(args.model_config_path)

# load model
model = models.load_model(vocab,
                          model_config)

# load runner config
runner_config = config.load_runner_config(args.runner_config_path)

# load runner
runner = runner.Runner(model,
                       runner_config)

# run training
while runner.run():

    # train model on training data
    runner.train(train_iterator)
    # evaluate model on validation data
    runner.eval(valid_iterator)

# evaluate model on test data
runner.eval(test_iterator)

# save tokenizer, vocab, model and runner
tokenizer.save(runner_config['run_path'])
vocab.save(runner_config['run_path'])
model.save(runner_config['run_path'])
runner.save(runner_config['run_path'])
