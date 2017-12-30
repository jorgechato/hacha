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


print('Loading data...')
data = Data(filename=args.data)
data.parse_data()
