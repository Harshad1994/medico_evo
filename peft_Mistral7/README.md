## Description of Solution

mistralai/Mistral-7B-v0.1 model is finetuned to do classificatin task.

Steps in solution

1. Instruct Dataset Preparation
2. Finetunning the model with LoRA PEFT technique
3. Merge LoRA adapters with original model and do inference

## How to RUN

System Requirements
Python >= 3.10  and system with GPU

1. Create virtual environment
2. Install requisite python libraries pip install -r requirements.txt
3. Run main.py
4. Hit the api endpoint with following curl command
```
curl -X 'POST' \
  'http://localhost:8080/classify' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "abstract": "This is my clinical abstract"
}'
```
   



