FROM python:3

ADD . /code
WORKDIR /code
RUN pip install -r requirements.txt

ENTRYPOINT python api.py -w /code/weights/text_generation.h5 -d /code/data/python.txt
