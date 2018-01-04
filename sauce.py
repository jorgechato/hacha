# -*- coding: utf-8 -*-
"""
main CLI

Author:  Jorge Chato
Repo:    github.com/jorgechato/hacha
Web:     jorgechato.com
"""
import sys
import os
import argparse
import datetime
import shutil

from sauce.data import Data
from sauce.utils import *
from sauce.LSTM.text_generation import Generate


CWD = os.getcwd() + "/"

parser = argparse.ArgumentParser()
parser.add_argument(
        "--neural",
        help     = "Type of neural network you want to train (Default: bidirectional).",
        metavar  = "NAME",
        default  = "bidirectional",
        )
parser.add_argument(
        "--epochs",
        help     = "Number of epochs to train the neural network (Default: 1).",
        metavar  = "NUMBER",
        default  = 20,
        type     = int,
        )
required = parser.add_argument_group('required arguments')
required.add_argument(
        "-w",
        "--weights",
        help     = "Add weights file to use the AI.",
        metavar  = "FILENAME",
        required = True,
        type     = str,
        )
required.add_argument(
        "-d",
        "--data",
        help     = "Add data file to train the AI.",
        metavar  = "FILENAME",
        required = True,
        type     = str,
        )


print_info()
args = parser.parse_args()

if len(sys.argv) == 1:
    print_argument_error()


if args.neural == "bidirectional":
    print('Loading data...')
    data = Data(
            filename   = args.data,
            maxlen     = 100,
            batch_size = 32,
            )
    data.parse_data()
elif args.neural == "text_generation":
    print('Loading data...')
    data = Data(
            filename   = args.data,
            maxlen     = 40,
            batch_size = 3,
            )
    data.parse_data()
    x, y = data.load_data()

    generate = Generate(
            maxlen       = data.maxlen,
            chars        = data.chars,
            char_indices = data.char_indices,
            text         = data.text,
            indices_char = data.indices_char,
            epochs       = args.epochs
            )

    print('Running...')
    generate.load_weights(args.weights)
    predict = generate.run()
    print(predict)
