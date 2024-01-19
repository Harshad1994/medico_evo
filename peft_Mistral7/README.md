## Description of Solution

mistralai/Mistral-7B-v0.1 [https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1] model is finetuned to do classificatin task.

Steps in solution

1. Instruct Dataset Preparation
2. Finetunning the model with LoRA PEFT technique
3. Merge LoRA adapters with original model and do inference

## How to RUN

System Requirements
Python >= 3.10  and system with GPU




The app is containerazed.
Simply build the provided Dockerfile and run

```
docker build -t my-mistral-app .

docker run -it my-mistral-app bash

docker run -p 8081:8081 my-mistral-app
```
Hit the api endpoint with following curl command / Python Client

```
```
curl -X 'POST' \
  'http://localhost:8081/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "abstract": "This is my clinical abstract"
}'
```

Python Client

```
import requests

url = 'http://localhost:8081/generate'
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}

data = {
    'abstract': 'This is my clinical abstract'
}

response = requests.post(url, headers=headers, json=data)

print(response.status_code)
print(response.json())

```
   



