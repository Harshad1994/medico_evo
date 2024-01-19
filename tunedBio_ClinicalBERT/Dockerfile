FROM python:3.10.8-slim
LABEL description="BioClinicalBERTapp"
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY app /app/
EXPOSE 8000
CMD ["python",  "main.py"]