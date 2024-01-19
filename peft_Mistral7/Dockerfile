FROM nvidia/cuda:12.3.1-base-ubuntu20.04

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

LABEL description="FineTunedMistral7B"

ENV APP_HOME=/app

COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

COPY app /app

WORKDIR $APP_HOME

CMD ["python",  "main_mistral.py"]