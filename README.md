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

1. Leveraging text embeddings models to convert abstract text intovector and then training a classical ML model on the vectorized text.
   Used OpenAI's "text-embedding-ada-002" text embeddings model.

2. Finetunning an Encoder only model for multi class classification.
   In this case we leverage Bio_ClinicalBERT  model already trained on clinical texts.
   This offers the advantage of domain specific knowledge already embedded in the model.
   Trained this model for multi class classification task.

3. Finetunning a Large Language Model for classification task
   In this case we finetuned a Mistral-7B-Instruct model for medical abstract classification task.

ReadMe file inside each solution directory shades more light on solution and how to use
   


