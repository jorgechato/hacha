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
import json
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
    batch_size = 128
    epochs     = 20
    maxlen     = 40
    """
    validation_split = 0.05 # 5% data for validation
    Epoch 30/30
    190270/190270 [==============================] - 101s 531us/step - loss: 1.3204 - acc: 0.5984 - val_loss: 1.5131 - val_acc: 0.5569
    """
    validation_split = 0.10 # 10% data for validation
    shuffle          = True

    chars = ""
    model = Sequential()

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.char_len = len(self.chars)

    def create(self):
        self.model.add(LSTM(128, input_shape=(self.maxlen, self.char_len)))
        self.model.add(Dense(self.char_len))
        self.model.add(Activation('softmax'))
        self.model = self.compile(self.model)
        self.batch_input_shape = (self.maxlen, self.char_len)

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

    def save(self, weights="weights/model.h5", json_model="out/model.json", history="out/history.p", info="out/info.json"):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(json_model, "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(weights)

        pickle.dump(self.history, open(history, "wb"))

        with open(info, "w", encoding='utf-8') as json_file:
            json_file.write(json.dumps({
                "char_indices":  self.char_indices,
                "indices_char":  self.indices_char,
                "char_len":      self.char_len,
                }, ensure_ascii=False))

    def load_weights(self, path):
        self.create_neurons()
        self.model.load_weights(path)
        self.history = pickle.load(open("out/history.p", "rb"))

    def load(self, weights="weights/model.h5", json_model="out/model.json", history="out/history.p", info="out/info.json"):
        """
        Load model, weights and neural network without the raw data.
        """
        json_file = open(json_model,'r')
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

        info_data = json.load(open(info,'r'))
        for key, value in info_data.items():
            setattr(self, key, value)
        self.indices_char = {int(k):v for k,v in self.indices_char.items()}

        self.model = loaded_model
        return graph

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
            x = prepare_input(text, self.char_len, self.char_indices, self.maxlen)

            preds = self.model.predict(x, verbose=0)[0]
            next_index = self.sample(preds, top_n=1)[0]
            next_char = self.indices_char[next_index]
            text = text[1:] + next_char
            completion += next_char

            if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
                return completion

    def predict(self, text, n=3):
        x = prepare_input(text, self.char_len, self.char_indices, self.maxlen)
        preds = self.model.predict(x, verbose=0)[0]
        next_indices = self.sample(preds, n)
        return [self.indices_char[idx] + self.prediction(text[1:] + self.indices_char[idx]) for idx in next_indices]
