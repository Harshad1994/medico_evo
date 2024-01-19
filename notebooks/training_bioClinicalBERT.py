

'''
Script to finetune BioClinical_BERT model on our classification problem

'''
import os
import pandas as pd
import numpy as np


from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler

import torch
from torch.optim import AdamW
import evaluate


from tqdm.auto import tqdm

train_file = r"../data/train.dat"

df = pd.read_csv(train_file, delimiter="\t",names=["labels","text"])

df["labels"]=df["labels"]-1

#convert df dataset to huggingface Dataset object
from datasets import Dataset, DatasetDict

df_s = Dataset.from_pandas(df)

# ds = ds.train_test_split(test_size=0.2, stratify_by_column="label")

base_model = 'emilyalsentzer/Bio_ClinicalBERT'


tokenizer = AutoTokenizer.from_pretrained(base_model)

def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=512,padding="max_length", truncation=True)

tokenized_datasets = df_s.map(tokenize_function, batched=True)


tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

ds = tokenized_datasets.train_test_split(test_size=0.3)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(ds["train"], shuffle=True, batch_size=16)
eval_dataloader = DataLoader(ds["test"], batch_size=16)

model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=5)

from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()

