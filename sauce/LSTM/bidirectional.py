# -*- coding: utf-8 -*-
"""
Trains a Bidirectional LSTM on the IMDB sentiment classification task.

Output after 4 epochs on CPU: ~0.8146

Author:  Jorge Chato
Repo:    github.com/jorgechato/hacha
Web:     jorgechato.com
"""
from __future__ import print_function
import numpy as np
import io

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from keras.preprocessing import sequence
from keras.datasets import mnist


class Sauce():
    # hyperparameters
    maxlen = 40
    step = 3

    sentences = []
    next_chars = []

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def parse_data(self):
        """
        Load the ASCII text for the file into memory and convert all of the
        characters to lowercase to reduce the vocabulary that the network must
        learn. Creating a map of each character to a unique integer.
        """
        # load the file with all the code
        self.text = io.open(self.filename, encoding='utf-8').read().lower()
        print('corpus length:', len(self.text))

        # list all the differents characters in the file project
        self.chars = sorted(list(set(self.text)))
        print('total chars:', len(self.chars))

        # map chars
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

        # cut the text in semi-redundant sequences of maxlen characters
        for i in range(0, len(self.text) - self.maxlen, self.step):
            self.sentences.append(self.text[i: i + self.maxlen])
            self.next_chars.append(self.text[i + self.maxlen])
        print('nb sequences:', len(self.sentences))

    def load_data(self):
        """
        Each training pattern of the network is comprised of 100 time steps of
        one character (X) followed by one character output (y). When creating
        these sequences, we slide this window along the whole file one
        character at a time, allowing each character a chance to be learned from
        the 100 characters that preceded it (except the first 100 characters of course).
        """
        pass


if __name__ == "__main__":
    print('Loading data...')
    sauce = Sauce(filename="./data/django.txt")
    sauce.parse_data()
    # (x_train, y_train), (x_test, y_test) = sauce.load_data()
