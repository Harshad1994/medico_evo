## Description of Solution


Finetunning a pre-trained Bio_ClinicalBERT [[emilyalsentzer/Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)] model for multi class classification.  
The base model offers the advantage of domain specific knowledge already embedded in the model.

## How to RUN

System Requirements
Python >= 3.10

1. Create virtual environment
2. Install requisite python libraries pip install -r requirements.txt
3. Run main.py
4. Hit the api endpoint with following curl command / Python Client
```
curl -X 'POST' \
  'http://localhost:8080/classify' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "abstract": "This is my clinical abstract"
}'
```

Python Client

```
import requests

url = 'http://localhost:8080/classify'
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
   




