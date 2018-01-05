# -*- coding: utf-8 -*-
"""
API to the plugin

Author:  Jorge Chato
Repo:    github.com/jorgechato/sauce
Web:     jorgechato.com
"""
import os
from flask import Flask
from flask import jsonify, request, abort, session
from flask.views import MethodView

from sauce.data import Data
from sauce.utils import *
from sauce.LSTM.text_generation import Generate

app = Flask(__name__)
global generate, graph
generate, graph = Generate().load()


class ModelLoader(MethodView):

    def __init__(self):
        """Initialize ModelLoader class."""
        pass

    def post(self):
        if not request.json:
            abort(400)

        print('Running...')
        seq = request.json["input"][-40:].lower()

        with graph.as_default():
            prediction = generate.predict(seq, 5)
            print(seq)
            print(prediction)
            print()

            return jsonify(
                    input = seq,
                    next  = prediction,
                    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))

    app.add_url_rule('/predict', view_func=ModelLoader.as_view('predict'))

    app.run(host='0.0.0.0', port=port)
