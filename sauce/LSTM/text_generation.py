# -*- coding: utf-8 -*-
"""
At least 20 epochs are required before the generated text starts sounding
coherent.

It is recommended to run this script on GPU

If you try this script on new data, make sure your corpus has at least ~100k
characters. ~1M is better.

Author:  Jorge Chato
Repo:    github.com/jorgechato/sauce
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

from sauce.data import prepare_input


model = Sequential()

# build the model: a single LSTM
class Generate():
    # hyperparameters
    batch_size  = 128
    epochs      = 20
    temperature = 1.0
    validation_split = 0.05 # 5% data for validation

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
                    validation_split = self.validation_split,
                    batch_size       = self.batch_size,
                    epochs           = self.epochs,
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

    def sample(self, preds, top_n=3):
        """
        This function allows us to ask our model what are the next n most
        probable characters.
        """
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        return heapq.nlargest(top_n, range(len(preds)), preds.take)

    def predict(self, text):
        """
        This function predicts next character until space is predicted (you can
        extend that to punctuation symbols). It does so by repeatedly preparing
        input, asking our model for predictions and sampling from them.
        """
        original_text = text
        generated = text
        completion = ''
        while True:
            x = prepare_input(text, self.chars, self.char_indices, self.maxlen)

            preds = model.predict(x, verbose=0)[0]
            next_index = self.sample(preds, top_n=1)[0]
            next_char = indices_char[next_index]
            text = text[1:] + next_char
            completion += next_char

            if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
                return completion

