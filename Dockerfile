FROM python:3.7

RUN pip install pylint twine sphinx sphinx-rtd-theme nbsphinx pandoc wandb

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
