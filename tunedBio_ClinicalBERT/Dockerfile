FROM python:3.10.8-slim

LABEL description="BioClinicalBERTapp"

ENV APP_HOME=/app

COPY requirements.txt /app/requirements.txt
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY app /app

WORKDIR $APP_HOME

CMD ["python",  "main.py"]