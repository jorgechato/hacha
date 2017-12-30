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


text = io.open("./data/django.txt", encoding='utf-8').read().lower()
print('corpus length:', len(text))
