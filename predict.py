# -*- coding: utf-8 -*-
"""
main CLI

Author:  Jorge Chato
Repo:    github.com/jorgechato/sauce
Web:     jorgechato.com
"""
import sys
import os
import argparse

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
elif args.neural == "text_generation":
    data = Data()
    chars        = data.get_chars(args.data)
    char_indices = data.get_char_indicies(chars)
    indices_char = data.get_indicies_char(chars)

    generate = Generate(
            maxlen       = 40,
            chars        = chars,
            char_indices = char_indices,
            indices_char = indices_char,
            )
    print('Loading weights...')
    generate.load_weights(args.weights)
    print('Running...')
    seq = "from django import time"[:40].lower()
    print(seq)
    print(generate.predict(seq, 5))
    print()
