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
        help     = "Type of neural network you want to train (Default: text_generation).",
        metavar  = "NAME",
        default  = "text_generation",
        )
required = parser.add_argument_group('required arguments')
required.add_argument(
        "-i",
        "--input",
        help     = "Input to predict",
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
    print('Loading weights...')
    generate = Generate()
    generate.load()
    print('Running...')
    seq = args.input[-40:].lower()
    print(seq)
    print(generate.predict(seq, 5))
    print()
