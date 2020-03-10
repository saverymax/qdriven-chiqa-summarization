"""
Module for data processing.

Includes classes for loading the data and generating examples
"""

import json
import tensorflow_datasets as tfds
import tensorflow as tf
from sklearn.utils import shuffle as sk_shuffle
import numpy as np
import random
import spacy


class Vocab():
    """
    Class to generate vocab for model
    """

    def train_tokenizer(self, tokenizer_filename, training_data, vocab_size):
        """
        Train the subword tokenizer
        """
        encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
           (sentence for abstract in training_data[1] for sentence in abstract), target_vocab_size=vocab_size)
        encoder.save_to_file(tokenizer_filename)

    def _load_tokenizer(self, tokenizer_filename):
        """
        Load the trained subword tokenizer
        """
        encoder = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_filename)
        return encoder

    def encode_data(self, tokenizer_filename, data):
        encoder = self._load_tokenizer(tokenizer_filename)
        encoded_question = [encoder.encode(question) for question in data[0]]
        encoded_answer = [[encoder.encode(sentence) for sentence in abstract] for abstract in data[1]]
        assert len(encoded_question) == len(encoded_answer)
        data = [encoded_question, encoded_answer]
        return data


class DataLoader():
    """
    Class to load data and generate train and validation datasets
    """

    def __init__(self, data_path, max_tok_q, max_sentences, max_tok_sent, dataset, summary_type):
        """
        Initiate loader
        """
        self.data_path = data_path
        self.max_tok_q = max_tok_q
        self.max_sentences = max_sentences
        self.max_tok_sent = max_tok_sent
        self.dataset = dataset
        self.summary_type = summary_type

    def split_data(self, data):
        """
        Shuffle and divide data up into training and validation sets
        """
        questions, documents, scores = data
        # Shuffle data:
        assert len(questions) == len(documents)
        assert len(questions) == len(scores)
        documents, questions, scores = sk_shuffle(documents, questions, scores, random_state=13)
        training_index = int(len(questions) * .8)
        x_train = [questions[:training_index], documents[:training_index]]
        y_train = scores[:training_index]
        x_val = [questions[training_index:], documents[training_index:]]
        y_val = scores[training_index:]

        return x_train, y_train, x_val, y_val

    def pad_data(self, data, data_type, padding_type='float32', test=False):
        """
        Method for padding data for training, validation, or inference.
        """
        if data_type == "question":
            padded_data = tf.keras.preprocessing.sequence.pad_sequences(data, padding='post', maxlen=self.max_tok_q)
        elif data_type == "scores":
            padded_data = tf.keras.preprocessing.sequence.pad_sequences(data, padding='post', dtype=padding_type, maxlen=self.max_sentences)
        elif data_type == "article":
            padded_data = []
            # Mask for sentences in document embedding
            #sentence_masks = []
            for doc in data:
                if len(doc) > self.max_sentences:
                    doc = doc[:self.max_sentences]
                    #sentence_masks.append(np.ones(len(doc), dype=bool))
                elif len(doc) < self.max_sentences:
                    # Add the sentences that are missing.
                    extra_docs = [[] for i in range(self.max_sentences - len(doc))]
                    doc.extend(extra_docs)
                    #sentence_mask = [False if d == [] else True for d in doc]
                    #sentence_masks.append(np.array(sentence_masks))
                    assert len(doc) == self.max_sentences
                padded_data.append(tf.keras.preprocessing.sequence.pad_sequences(doc, padding='post', maxlen=self.max_tok_sent))

        # For asserting correct padding:
        if test:
            if data_type == "question":
                for q in padded_data:
                    assert len(q) == self.max_tok_q, q
            if data_type == "article":
                #assert len(sentence_masks) == len(padded_data), len(sentence_masks)
                #for m in sentence_masks:
                #    assert len(m) == self.max_sentences, len(m)
                for doc in padded_data:
                    assert isinstance(doc, np.ndarray), type(doc)
                    assert len(doc) == self.max_sentences, len(doc)
                    for sent in doc:
                        assert isinstance(sent, np.ndarray), type(sent)
                        assert len(sent) == self.max_tok_sent, len(sent)
                        assert isinstance(sent[0], np.int32), (sent[0], type(sent[0]))
            elif data_type == "scores":
                for doc in padded_data:
                    assert isinstance(doc, np.ndarray), type(doc)
                    assert len(doc) == self.max_sentences, len(doc)
            # Convert abstracts to np array here
            # pad_sequences converts lists of lists to np arrays, which is the form the questions
            # and sentences are in.
            # Reshape y_train and val from 2d > 3d
        if data_type == "scores":
            padded_data = np.reshape(padded_data, padded_data.shape + (1,))
            return padded_data
        elif data_type == "article":
            return np.array(padded_data)
        else:
            return padded_data

    def load_data(self, mode, tag_sentences):
        """
        Open the data and split into train/val.
        """
        questions = []
        documents = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            asumm_data = json.load(f)

        if mode == "train":
            scores = []
            for q_id in asumm_data:
                questions.append(asumm_data[q_id]['question'])
                documents.append(asumm_data[q_id]['sentences'])
                scores.append(asumm_data[q_id]['labels'])
            return questions, documents, scores

        if mode == "infer":
            summaries = []
            nlp = spacy.load('en_core_web_sm')
            if self.dataset == "chiqa" or self.dataset=="medinfo":
                question_ids = []
                # There are multiple summary tasks withing the chiqa dataset: single and multi doc require different processing.
                # Handle the single answer -> summary case:
                if "single" in self.summary_type:
                    for q_id in asumm_data:
                        question_ids.append(q_id)
                        question = asumm_data[q_id]['question']
                        questions.append(question)
                        summary = asumm_data[q_id]['summary']
                        tokenized_art = nlp(asumm_data[q_id]['articles'])
                        tokenized_summ = nlp(summary)
                        # Split sentences and tag with s if option included for pointer generator
                        if tag_sentences:
                            summary = " ".join(["<s> {s} </s>".format(s=s.text.strip()) for s in tokenized_summ.sents])
                        summaries.append(summary)
                        article_sentences = [s.text.strip() for s in tokenized_art.sents]
                        documents.append(article_sentences)
            return questions, documents, summaries, question_ids


if __name__ == "__main__":
    # Locally testing data classes:
    max_tok_q = 20
    max_sentences = 75
    max_tok_sent = 20
    hidden_dim = 256
    dataset = "bioasq"
    tag_sentences = False
    mode = "train"
    data_loader = DataLoader("/data/saveryme/asumm/asumm_data/training_data/bioasq_abs2summ_sent_classification_training.json", max_tok_q, max_sentences, max_tok_sent, dataset)
    data = data_loader.load_data(mode, tag_sentences)
    x_train, y_train, x_val, y_val = data_loader.split_data(data)
    # x_train[0] = x_train[0][:3]
    # x_train[1] = x_train[1][:3]
    # x_val[0] = x_val[0][:3]
    # x_val[1] = x_val[1][:3]
    # y_train = y_train[:3]
    # y_val = y_val[:3]

    # print("Text!")
    print("Questions:\n", x_train[0][0])
    print("Sentences:\n", x_train[1][0])
    print("encoding data")
    x_train = Vocab().encode_data("medsumm_bioasq_abs2summ/tokenizer", x_train)

    sent_cnt = 0
    sent_len = 0
    for doc in x_train[1]:
        for sentence in doc:
            sent_cnt += 1
            sent_len += len(sentence)

    x_val = Vocab().encode_data("medsumm_bioasq_abs2summ/tokenizer", x_val)
    print("padding data")
    x_train[0] = data_loader.pad_data(x_train[0], data_type="question", test=True)
    x_train[1] = data_loader.pad_data(x_train[1], data_type="article", test=True)
    x_val[0] = data_loader.pad_data(x_val[0], data_type="question", test=True)
    x_val[1] = data_loader.pad_data(x_val[1], data_type="article", test=True)
    y_train = data_loader.pad_data(y_train, data_type="scores", test=True)
    y_val = data_loader.pad_data(y_val, data_type="scores", test=True)
    assert x_train[0].shape == (30532, max_tok_q), x_train[0].shape
    assert x_train[1].shape == (30532, max_sentences, max_tok_sent), x_train[1].shape
    assert x_val[0].shape == (7634, max_tok_q), x_val[0].shape
    assert x_val[1].shape == (7634, max_sentences, max_tok_sent), x_val[1].shape
    assert y_train.shape == (30532, max_sentences, 1), y_train.shape
    assert y_val.shape == (7634, max_sentences, 1), y_val.shape
    print("Avg. subwords/sentence:", sent_len / sent_cnt)
