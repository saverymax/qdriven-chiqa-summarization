"""
Module to make vocab file for pointer generator network. Code modeled from make_datafiles.py
in cnn-dailymail processing repository.

File will have format
word1 n_occurences
word2 n_occurences
...
wordn n_occurences

To run:
python make_asumm_pg_vocab.py --vocab_path=bioasq_abs2summ_vocab --data_file=../../data_process/data/bioasq_abs2summ_training_data_without_question.json
"""

from collections import Counter
import json
import argparse

from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

def get_args():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description="Arguments for data exploration")
    parser.add_argument("--vocab_path",
                        dest="vocab_path",
                        help="Path to create vocab file")
    parser.add_argument("--data_file",
                        dest="data_file",
                        help="Path to load data to make vocab")
    return parser


def make_vocab(vocab_counter, vocab_file, VOCAB_SIZE, article, abstract, tokenizer):
    """
    For each page/summary pair, tokenize on spaces, do a word count, and save tokens to file
    """
    art_tokens = [t.text.strip() for t in tokenizer(article)]
    abs_tokens = [t.text.strip() for t in tokenizer(abstract)]
    tokens = art_tokens + abs_tokens
    tokens = [t for t in tokens if t != "" and t != "<s>" and t != "</s>"]
    vocab_counter.update(tokens)


def load_data(data_file, vocab_path):
    """
    Load data and tokenize each file
    """
    VOCAB_SIZE = 200000
    with open(data_file, "r", encoding="utf-8") as f:
        training_data = json.load(f)
    print("Writing vocab file...")
    vocab_file =  open(vocab_path, 'w', encoding="utf-8")
    vocab_counter = Counter()
    # Initiate spacy tokenizer without bells and whistles
    nlp = English()
    tokenizer = Tokenizer(nlp.vocab)

    for url, topic in training_data.items():
        article = topic['articles']
        abstract = topic['summary']
        make_vocab(vocab_counter, vocab_file, VOCAB_SIZE, article, abstract, tokenizer)
    # After updating counter, write counts/words
    print("20 most common words:", vocab_counter.most_common(20))
    for word, count in vocab_counter.most_common(VOCAB_SIZE):
        vocab_file.write(word + ' ' + str(count) + '\n')

    vocab_file.close()
    print("Finished writing vocab file")


if __name__ == "__main__":
    args = get_args().parse_args()
    load_data(args.data_file, args.vocab_path)
