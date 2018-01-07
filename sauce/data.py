# -*- coding: utf-8 -*-
"""
Parse and prepare data

Author:  Jorge Chato
Repo:    github.com/jorgechato/sauce
Web:     jorgechato.com
"""
from __future__ import print_function
import numpy as np
import io


class Data():
    # hyperparameters
    maxlen = 40
    step   = 32

    sentences  = []
    next_chars = []

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_chars(self, filename=None):
        if not filename:
            filename = self.filename
        # load the file with all the code
        self.text = io.open(filename, encoding='utf-8').read().lower()
        print('corpus length:', len(self.text))

        # list all the differents characters in the file project
        self.chars = sorted(list(set(self.text)))
        print('total chars:', len(self.chars))
        return self.chars

    def get_char_indicies(self, chars=None):
        if not chars:
            chars = self.chars

        return dict((c, i) for i, c in enumerate(chars))

    def get_indicies_char(self, chars=None):
        if not chars:
            chars = self.chars

        return dict((i, c) for i, c in enumerate(chars))

    def parse_data(self):
        """
        Load the ASCII text for the file into memory and convert all of the
        characters to lowercase to reduce the vocabulary that the network must
        learn. Creating a map of each character to a unique integer.
        """
        self.get_chars()
        # map chars
        self.char_indices = self.get_char_indicies()
        self.indices_char = self.get_indicies_char()

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
        print('Vectorization...')
        x = np.zeros((len(self.sentences), self.maxlen, len(self.chars)), dtype=np.bool)
        y = np.zeros((len(self.sentences), len(self.chars)), dtype=np.bool)
        for i, sentence in enumerate(self.sentences):
            for t, char in enumerate(sentence):
                x[i, t, self.char_indices[char]] = 1
            y[i, self.char_indices[self.next_chars[i]]] = 1

        return x, y


def prepare_input(text, chars, char_indices, maxlen):
    """
    Prepare Input to predict the next word. Convert a text input to a 3D
    numpy array (required as the LSTM model is build)
    """
    x = np.zeros((1, maxlen, chars))
    for t, char in enumerate(text):
        x[0, t, char_indices[char]] = 1.

    return x
