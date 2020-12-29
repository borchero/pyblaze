FROM python:3.7

RUN apt-get update && apt-get install -y pandoc
RUN pip install pylint twine sphinx sphinx-rtd-theme nbsphinx wandb ipython

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
