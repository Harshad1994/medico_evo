# medico_evo
Identify the disease category basis the clinical/medical abstracts.

### Dataset
train.dat and test.dat files are present in data folder.

Columns
1. disease category --> 1,2,3,4,5

2. abstract --> text


Classes / Categories
```
disease_categories = {
    1: "digestive system diseases",
    2: "cardiovascular diseases",
    3: "neoplasms",
    4: "nervous system diseases",
    5: "general pathological conditions"
}
```


### Solutions

1. Leveraging text embeddings models to convert abstract text into vector and then training a classical ML model on the vectorized text.
   Used OpenAI's "text-embedding-ada-002" text embeddings model.

2. Finetunning an Encoder only model for multi class classification.
   In this case we leverage Bio_ClinicalBERT  model already trained on clinical texts.
   This offers the advantage of domain specific knowledge already embedded in the model.
   Trained this model for multi class classification task.

3. Finetunning a Large Language Model for classification task
   In this case we finetuned a Mistral-7B-Instruct model for medical abstract classification task.

# Solution 1:
    Solution approach and details can be seen in notebooks/model_with_embeddings.ipynb
    The embeddings lack the clinical domain understaing and hence model does not do so well.

# Soution 2

Finetunning a pre-trained Bio_ClinicalBERT [[emilyalsentzer/Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)] model for multi class classification.  
The base model offers the advantage of domain specific knowledge already embedded in the model.

### How to RUN

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
   
   
# Soution 3:

mistralai/Mistral-7B-v0.1 [https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1] model is finetuned to do classificatin task.

Steps in solution

1. Instruct Dataset Preparation
2. Finetunning the model with LoRA PEFT technique
3. Merge LoRA adapters with original model and do inference

### How to RUN

The app is containerazed.
Simply build the provided Dockerfile and run

```
docker build -t my-mistral-app .

docker run --gpus all -p 8081:8081 my-mistral-app
```
Hit the api endpoint with following curl command / Python Client

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

Response Json
```
{
    'res_id':'id',
    'category' : 'disease category',
    'category_index':1,
    'confidence':0.7
}

```




