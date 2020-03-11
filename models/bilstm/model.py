"""
Model module for constructing the tensorflow graph for the LSTM sentence classifier
"""

import tensorflow as tf
from tensorflow.keras import layers

class SentenceClassificationModel():
    """
    Class for model selecting sentences from relevant documents for answer summarization
    """

    def __init__(self, vocab_size, batch_size, hidden_dim, dropout, max_tok_q, max_sentences, max_tok_sent):
        """
        Initiate the mode
        """
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.max_tok_q = max_tok_q
        self.max_tok_sent = max_tok_sent
        self.max_sentences = max_sentences
        self.dropout = dropout

    def build_binary_model(self):
        """
        Construct the graph of the model
        """
        question_input = tf.keras.Input(shape=(self.max_tok_q, ), name='q_input')
        abstract_input = tf.keras.Input(shape=(self.max_sentences, self.max_tok_sent, ), name='abs_input')
        # NOT USING MASKING DUE TO CUDNN ERROR: https://github.com/tensorflow/tensorflow/issues/33148
        x1 = layers.Embedding(input_dim=self.vocab_size, output_dim=self.hidden_dim, mask_zero=False)(question_input)
        x1 = layers.Bidirectional(layers.LSTM(self.hidden_dim, dropout=self.dropout, kernel_regularizer=tf.keras.regularizers.l2(0.01)), input_shape=(self.max_tok_q, self.hidden_dim),  name='q_bilstm')(x1)

        # Apply embedding to every sentence
        x2 = layers.TimeDistributed(layers.Embedding(input_dim=self.vocab_size, output_dim=self.hidden_dim, input_length=self.max_tok_sent, mask_zero=False), input_shape=(self.max_sentences, self.max_tok_sent))(abstract_input)
        # Apply lstm to every sentence embedding
        x2 = layers.TimeDistributed(layers.Bidirectional(layers.LSTM(self.hidden_dim, dropout=self.dropout, kernel_regularizer=tf.keras.regularizers.l2(0.01))), input_shape=(self.max_sentences, self.max_tok_sent, self.hidden_dim), name='sentence_distributed_bilstms')(x2)
        # Make lstm of document representation:
        # I could also just take this document representation and concatenate it to the single sentence representation, but I don't.
        x2 = layers.Bidirectional(layers.LSTM(self.hidden_dim, return_sequences=True, dropout=self.dropout, kernel_regularizer=tf.keras.regularizers.l2(0.01)), input_shape=(self.max_sentences, self.hidden_dim * 2), name='document_bilstm')(x2)
        # Combine question and document
        x3 = layers.RepeatVector(self.max_sentences)(x1)
        x4 = layers.concatenate([x2, x3])

        # If using integers as class labels, 1 target label can be provided be example (not 1 hot) and the number of labels can be defined here
        sent_output = layers.Dense(2, activation='sigmoid', name='sent_output', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x4)

        model = tf.keras.Model(inputs=[question_input, abstract_input], outputs=sent_output)
        model.summary()
        model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4)
              )

        return model
