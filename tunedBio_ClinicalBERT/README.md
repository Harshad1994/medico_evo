## Description of Solution


Finetunning a pre-trained Bio_ClinicalBERT [[emilyalsentzer/Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)] model for multi class classification.  
The base model offers the advantage of domain specific knowledge already embedded in the model.

## How to RUN

The app is containerazed.
Simply build the provided Dockerfile and run

```
docker build -t my-bert-app .

docker run -p 8080:8080 my-bert-app
```
Hit the api endpoint with following curl command / Python Client

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


Response Json
```
{
    'res_id':'id',
    'category' : 'disease category',
    'category_index':1,
    'confidence':0.7
}

```
   




