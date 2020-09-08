import argparse

#our modules
import tokenizer
import data
import datasets
import vocab


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
args = parser.parse_args()

# get tokenizer
tokenizer = tokenizer.Tokenizer(args.tokenize_fn,
                                args.lower,
                                args.sos_token,
                                args.eos_token,
                                args.max_length)

# load data and create vocabulary
dataset = data.LanguageModelingDataset(args.train_data_path,
                                       args.valid_data_path,
                                       args.test_data_path,
                                       tokenizer,
                                       min_freq=args.min_freq,
                                       max_size=args.max_size,
                                       unk_token=args.unk_token,
                                       pad_token=args.pad_token,
                                       special_tokens=args.special_tokens)

# get data iterators
train_iterator, valid_iterator, test_iterator = dataset.load_iterators(args.batch_size, args.sequence_length)
