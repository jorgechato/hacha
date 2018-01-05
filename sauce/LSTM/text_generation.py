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
import heapq
import pickle
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop

from sauce.data import prepare_input


# build the model: a single LSTM
class Generate():
    # hyperparameters
    batch_size       = 128
    epochs           = 20
    validation_split = 0.05 # 5% data for validation
    shuffle          = True

    model = Sequential()

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def create(self):
        self.model.add(LSTM(128, input_shape=(self.maxlen, len(self.chars))))
        self.model.add(Dense(len(self.chars)))
        self.model.add(Activation('softmax'))
        self.model = self.compile(self.model)

    def compile(self, model):
        optimizer = RMSprop(lr=0.01)
        model.compile(
                loss      = 'categorical_crossentropy',
                optimizer = optimizer,
                metrics   = ['accuracy'],
                )

        return model

    def train(self, x, y):
        # train the model, output generated text after each iteration
        self.history = self.model.fit(
                x,
                y,
                validation_split = self.validation_split,
                batch_size       = self.batch_size,
                epochs           = self.epochs,
                shuffle          = True,
                ).history

        start_index = random.randint(0, len(self.text) - self.maxlen - 1)

        self.save()

    def save(self, weights="weights/model.h5", json="out/model.json", history="out/history.p"):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(json, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(weights)
        pickle.dump(self.history, open(history, "wb"))

    def load_weights(self, path):
        self.create_neurons()
        self.model.load_weights(path)
        self.history = pickle.load(open("out/history.p", "rb"))

    def load(self, weights="weights/model.h5", json="out/model.json", history="out/history.p"):
        """
        Load model, weights and neural network without the raw data.
        """
        json_file = open(json,'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        #load woeights into new model
        loaded_model.load_weights(weights)
        print("Loaded Model from disk")

        #compile and evaluate loaded model
        loaded_model = self.compile(loaded_model)

        graph = tf.get_default_graph()
        self.history = pickle.load(open(history, "rb"))

        return loaded_model, graph

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

    def prediction(self, text):
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

            preds = self.model.predict(x, verbose=0)[0]
            next_index = self.sample(preds, top_n=1)[0]
            next_char = self.indices_char[next_index]
            text = text[1:] + next_char
            completion += next_char

            if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
                print(completion)
                return completion

    def predict(self, text, n=3):
        x = prepare_input(text, self.chars, self.char_indices, self.maxlen)
        preds = self.model.predict(x, verbose=0)[0]
        next_indices = self.sample(preds, n)
        return [self.indices_char[idx] + self.prediction(text[1:] + self.indices_char[idx]) for idx in next_indices]
