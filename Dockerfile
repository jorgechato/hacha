FROM python:3

ADD . /code
WORKDIR /code
RUN pip install -r requirements.txt

ENTRYPOINT python train.py --data data/train.txt
