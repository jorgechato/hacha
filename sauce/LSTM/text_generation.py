# -*- coding: utf-8 -*-
"""
At least 20 epochs are required before the generated text starts sounding
coherent.

It is recommended to run this script on GPU

If you try this script on new data, make sure your corpus has at least ~100k
characters. ~1M is better.

Author:  Jorge Chato
Repo:    github.com/jorgechato/hacha
Web:     jorgechato.com
"""
from __future__ import print_function
import numpy as np
import random
import sys
import io

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import keras.preprocessing.text


model = Sequential()

# build the model: a single LSTM
class Generate():
    # hyperparameters
    batch_size  = 128
    epochs      = 1
    temperature = 1.0

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def create_neurons(self):
        model.add(LSTM(128, input_shape=(self.maxlen, len(self.chars))))
        model.add(Dense(len(self.chars)))
        model.add(Activation('softmax'))


    def compile(self):
        self.create_neurons()

        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)


    def train(self, x, y):
        # train the model, output generated text after each iteration
        for iteration in range(1, 10):
            print()
            print('-' * 50)
            print('Iteration', iteration)
            model.fit(
                    x,
                    y,
                    batch_size = self.batch_size,
                    epochs     = self.epochs,
                    )

            start_index = random.randint(0, len(self.text) - self.maxlen - 1)

            for diversity in [0.2, 0.5, 1.0, 1.2]:
                print()
                print('----- diversity:', diversity)

                generated = ''
                sentence = self.text[start_index: start_index + self.maxlen]
                generated += sentence
                print('----- Generating with seed: "' + sentence + '"')
                sys.stdout.write(generated)

                for i in range(400):
                    x_pred = np.zeros((1, self.maxlen, len(self.chars)))
                    for t, char in enumerate(sentence):
                        x_pred[0, t, self.char_indices[char]] = 1.

                    preds = model.predict(x_pred, verbose=0)[0]
                    next_index = self.sample(preds, diversity)
                    next_char = self.indices_char[next_index]

                    generated += next_char
                    sentence = sentence[1:] + next_char

                    sys.stdout.write(next_char)
                    sys.stdout.flush()
                print()
        model.save_weights("weights/text_generation.h5")


    def sample(self, preds, temperature = None):
        if not temperature:
            temperature = self.temperature
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def load_weights(self, path):
        self.create_neurons()
        model.load_weights(path)

    def run(self):
        text = np.array(["from django import routes"])
        tk = keras.preprocessing.text.Tokenizer(
                nb_words = 2000,
                filters  = keras.preprocessing.text.base_filter(),
                lower    = True,
                split    = " ",
                )
        tk.fit_on_texts(text)

        return model.predict(
                np.array(
                    tk.texts_to_sequences(text)
                    )
                )
