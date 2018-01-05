# Sauce

A recurrent neural network & LSTM AI to autocomplete code.

### Install
#### Local
```zsh
$ pip install -r requirements.txt
```
#### Docker
```zsh
$ docker build -t sauce .

$ docker run -d \
-v ${PWD}/data:/code/data \
-v ${PWD}/out:/code/out \
-v ${PWD}/weights:/code/weights \
--name sauce-train sauce
```
### Get data
Concatenate all files you have written splitted by language in a single file. You
can use the following command:
```zsh
$ cat <projects-folder>/**/*.py > ./data/python.txt
```
### Train
```zsh
$ python train.py --data <file>
# or to show the full options
$ python train.py -h
```
### Folder structure
```zsh
.
├── Dockerfile
├── README.md
├── requirements.txt
├── sauce
│   ├── data.py
│   ├── __init__.py
│   ├── LSTM
│   │   ├── bidirectional.py
│   │   ├── __init__.py
│   │   └── text_generation.py
│   └── utils.py
├── train.py
├── predict.py
├── api.py
├── data
├── out
└── weights
```
