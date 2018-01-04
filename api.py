# -*- coding: utf-8 -*-
"""
API to the plugin

Author:  Jorge Chato
Repo:    github.com/jorgechato/sauce
Web:     jorgechato.com
"""
import sys
import os
import argparse
from flask import Flask
from flask import jsonify, request, abort

from sauce.data import Data
from sauce.utils import *
from sauce.LSTM.text_generation import Generate

app = Flask(__name__)
generate = None


@app.route("/generate", methods=["POST"])
def generate_word():
    if not request.json:
        abort(400)

    print('Running...')
    prediction = generate.predict(request.json["input"].lower())
    return jsonify(next=prediction)

@app.route("/generate/reload/<data>/<weights>", methods=["POST"])
def load_generate_config(data, weights):
    chars        = Data().get_chars(data)
    char_indices = Data().get_char_indicies(chars)

    generate = Generate(
            maxlen       = 40,
            chars        = chars,
            char_indices = char_indices,
            )
    print('Loading weights...')
    generate.load_weights(weights)
    return


if __name__ == '__main__':
    CWD = os.getcwd() + "/"

    parser = argparse.ArgumentParser()
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

    load_generate_config(args.data, args.weights)
    app.run(host= '0.0.0.0',debug=True)
