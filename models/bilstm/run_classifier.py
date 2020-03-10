"""
Runner script to train a simple lstm sentence classifier,
using best rouge score for a sentence in the article compared to each sentence in the summary
as the y labels.

Should I create labels with rouge or with n sentences selected based on highes score.
"""

import time
from absl import app, flags
import logging
import os
import sys
import json
import re

import numpy as np
import tensorflow as tf

from data import Vocab, DataLoader
from model import SentenceClassificationModel

FLAGS = flags.FLAGS

# Paths
flags.DEFINE_string('data_path', '', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')
flags.DEFINE_string('mode', 'train', 'Select train/infer')
flags.DEFINE_string('exp_name', '', 'Name for experiment. Tensorboard logs and models will be saved in a directories under this one')
flags.DEFINE_string('tensorboard_log', '', 'Name of specific run for tb to log to. The experiment name will be the parent directory')
flags.DEFINE_string('model_image_path', '', "Path to save image of model graph")
flags.DEFINE_string('model_path', '', "Path to save model checkpoint")
flags.DEFINE_string('dataset', '', "The dataset used for inference, used to specify how to load the data, e.g., medinfo")
flags.DEFINE_string('summary_type', '', "The summary task within the chiqa dataset. The multi and single document tasks require different data handling")
flags.DEFINE_string('prediction_file', '', "File to save predictions to be used for generative model")
flags.DEFINE_string('eval_file', '', "File to save preds for direct evaluation")

# Tokenizer training
flags.DEFINE_string('tokenizer_path', '', 'Path to save tokenizer once trained')
flags.DEFINE_boolean('train_tokenizer', False, "Flag to train a new tokenizer on the training corpus")

# Data processing
flags.DEFINE_boolean("tag_sentences", False, "For use with mode=infer. Tag the article sentences with <s> and </s>, when using pointer generator network as second step, if the data has not already been tagged")

# Hyperparameters and such
flags.DEFINE_boolean('binary_model', False, "Flag to use binary model or regression model")
flags.DEFINE_integer('vocab_size', 2**15, 'Size of subword vocabulary. Not sure if this is what I am definitely going to do. May be better to use pretrained embeddings')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('n_epochs', 10, 'Max number of epochs to run')
flags.DEFINE_integer('max_tok_q', 100, 'max number of tokens for question')
flags.DEFINE_integer('max_sentences', 10, 'max number of sentences')
flags.DEFINE_integer('max_tok_sent', 100, 'max number of subword tokens/sentence')
flags.DEFINE_integer('hidden_dim', 128, 'dimension of lstm')
flags.DEFINE_float('dropout', .2, 'dropout proportion')
flags.DEFINE_float('decision_threshold', .3, "Threshold for selecting relevant sentences during inference. No current implementation")
flags.DEFINE_integer('top_k_sent', 10, "Number of sentences to select.")


def run_training(model, x_train, y_train, x_val, y_val):
    """
    Run training in loop for n_epochs
    """
    logging.info("Beginning training\n")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=FLAGS.tensorboard_log, update_freq=5000, profile_batch=0)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='{0}/{1}'.format(FLAGS.tensorboard_log, FLAGS.model_path),
        # overwrite current checkpoint if `val_loss` has improved.
        save_best_only=True,
        monitor='val_loss',
        save_freq='epoch',
        verbose=0)

    model.fit({'q_input': x_train[0], 'abs_input': x_train[1]}, {'sent_output': y_train},
              shuffle=True,
              epochs=FLAGS.n_epochs,
              validation_data=(x_val, y_val),
              callbacks=[tensorboard_callback, checkpoint_callback])
    logging.info("Training completed!\n")


def run_inference(model, data, threshold, top_k_sent, binary_model):
    """
    Given a set of documents and a question, predict relevant sentences
    """
    predictions = model.predict({'q_input': data[0], 'abs_input': data[1]})
    logging.info("Predictions shape: {}".format(predictions.shape))
    #print(predictions)
    if binary_model:
        # Select preds for just label 1
        reduced_preds = predictions[:, :, 1]
        reduced_preds = tf.squeeze(reduced_preds)
        filtered_predictions = tf.math.top_k(reduced_preds, k=top_k_sent, sorted=True)
        # Using threshold
        # filtered_predictions = np.argwhere(predictions > threshold)
        #print(filtered_predictions)
    else:
        reduced_preds = tf.squeeze(predictions)
        filtered_predictions = tf.math.top_k(reduced_preds, k=top_k_sent)
    return filtered_predictions


def save_predictions(predictions, data, binary_model):
    """
    Get the indicies of sentences predicted to be above rouge threshold and save to json.
    """
    pred_dict = {}
    # Also save the data in format for direct evaluation
    questions = []
    ref_summaries = []
    gen_summaries = []
    question_ids = []
    pred_dict = {}
    q_cnt = 0
    for p, indices in zip(predictions.values, predictions.indices):
        question = data[0][q_cnt]
        question_id = data[3][q_cnt]
        pred_dict[question_id] = {}
        # Get the human generated summary
        ref_summary = data[2][q_cnt]
        pred_dict[question_id]['summary'] = ref_summary
        pred_dict[question_id]['question'] = question
        # Remove any predicted indices that are out of the article's range
        indices = indices.numpy()[indices.numpy() < len(data[1][q_cnt])]
        sentences = " ".join(list(np.array(data[1][q_cnt])[indices]))
        pred_dict[question_id]['articles'] = sentences
        pred_dict[question_id]['predicted_score'] = p.numpy().tolist()
        # Format for rouge evaluation
        questions.append(question)
        question_ids.append(question_id)
        # Remove any sentence tags from data that will be evaluated directly and not passed to pg
        ref_summary = ref_summary.replace("<s>", "")
        ref_summary = ref_summary.replace("</s>", "")
        ref_summaries.append(ref_summary)
        gen_summaries.append(sentences)
        q_cnt += 1

    predictions_for_eval = {'question_id': question_ids, 'question': questions, 'ref_summary': ref_summaries, 'gen_summary': gen_summaries}

    with open(FLAGS.prediction_file, "w", encoding="utf-8") as f:
        json.dump(pred_dict, f, indent=4)

    with open(FLAGS.eval_file, "w", encoding="utf-8") as f:
        json.dump(predictions_for_eval, f, indent=4)


def main(argv):
    """
    Main function for running sentence classifier
    """
    logging.info("Num GPUs Available: {}\n".format(len(tf.config.experimental.list_physical_devices('GPU'))))
    logging.basicConfig(filename="{}/medsumm.log".format(FLAGS.exp_name), filemode='w', level=logging.DEBUG)
    logging.info("TODO: Add masking to padded sentences, which will probably mask whole inputs. So figure out how to do that without getting error. This is addressed here: https://github.com/tensorflow/tensorflow/issues/33069")
    logging.info("Initiating sentence classication model...\n")
    logging.info("Loading data:")
    data_loader = DataLoader(FLAGS.data_path, FLAGS.max_tok_q, FLAGS.max_sentences, FLAGS.max_tok_sent, FLAGS.dataset, FLAGS.summary_type)
    # Returns tuple
    data = data_loader.load_data(FLAGS.mode, FLAGS.tag_sentences)
    logging.info("Questions: {}".format(data[0][0:2]))
    logging.info("Sentences: {}".format(data[1][0][:2]))
    if FLAGS.mode == "train":
        x_train, y_train, x_val, y_val = data_loader.split_data(data)
        logging.info("Questions")
        logging.info(x_train[0][:2])
        logging.info("Sentences")
        logging.info(x_train[1][:2])

    vocab_processor = Vocab()
    if FLAGS.train_tokenizer:
        logging.info("Training tokenizer\n")
        vocab_processor.train_tokenizer(FLAGS.tokenizer_path, x_train, FLAGS.vocab_size)
    # Once trained, get the subword tokenizer and encode the data
    logging.info("Encoding text")
    if FLAGS.mode == "train":
        logging.info("Encoding data")
        x_train = vocab_processor.encode_data(FLAGS.tokenizer_path, x_train)
        x_val = vocab_processor.encode_data(FLAGS.tokenizer_path, x_val)
        logging.info("Padding encodings")
        x_train[0] = data_loader.pad_data(x_train[0], data_type="question", test=True)
        x_train[1] = data_loader.pad_data(x_train[1], data_type="article", test=True)
        x_val[0] = data_loader.pad_data(x_val[0], data_type="question", test=True)
        x_val[1] = data_loader.pad_data(x_val[1], data_type="article", test=True)
        if FLAGS.binary_model:
            padding_type = "int32"
        else:
            padding_type = "float32"
        y_train = data_loader.pad_data(y_train, data_type="scores", padding_type=padding_type, test=True)
        y_val = data_loader.pad_data(y_val, data_type="scores", padding_type=padding_type, test=True)
        logging.info("Data shape:")
        logging.info("x_train questions: {}".format(x_train[0].shape))
        logging.info("x_train documents: {}".format(x_train[1].shape))
        logging.info("x_val questions: {}".format(x_val[0].shape))
        logging.info("x_val documents: {}".format(x_val[1].shape))
        logging.info("y_train: {}".format(y_train.shape))
        logging.info("y_val: {}".format(y_val.shape))

        logging.info("Question encoding")
        logging.info(x_train[0][:2])
        logging.info("Sentence encoding")
        logging.info(x_train[1][:2])
        logging.info("Rouge scores:")
        logging.info(y_train[0][:2])

        model = SentenceClassificationModel(
            FLAGS.vocab_size, FLAGS.batch_size, FLAGS.hidden_dim, FLAGS.dropout,
            FLAGS.max_tok_q, FLAGS.max_sentences, FLAGS.max_tok_sent,
            FLAGS.model_image_path
            )

        if FLAGS.binary_model:
            model = model.build_binary_model()
            run_training(model, x_train, y_train, x_val, y_val)
        else:
            model = model.build_model()
            run_training(model, x_train, y_train, x_val, y_val)

    if FLAGS.mode == "infer":
        logging.info("TOD: handle all chiqa dataset data processing cases")
        logging.info("Currently padding for inference. Is this necessary?")
        encoded_data = vocab_processor.encode_data(FLAGS.tokenizer_path, data)
        padded_data = []
        padded_data.append(data_loader.pad_data(encoded_data[0], "question"))
        padded_data.append(data_loader.pad_data(encoded_data[1], "article"))
        logging.info("N questions: {}".format(len(padded_data[0])))
        logging.info("N documents: {}".format(len(padded_data[1])))
        logging.info("Question encoding")
        logging.info(padded_data[0][:2])
        logging.info("Sentence encoding")
        logging.info(padded_data[1][:2])

        logging.info("Loading model")
        model = tf.keras.models.load_model(FLAGS.model_path)
        predictions = run_inference(model, padded_data, FLAGS.decision_threshold, FLAGS.top_k_sent, FLAGS.binary_model)
        save_predictions(predictions, data, FLAGS.binary_model)


if __name__ == "__main__":
    app.run(main)
