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


class Classificate:
    pass
