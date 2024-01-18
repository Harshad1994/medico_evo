
#Script to finetune mistral 7b on our own instruct dataset


from datasets import load_dataset

train_dataset = load_dataset('json', data_files='../data/train_1.jsonl', split='train')
eval_dataset = load_dataset('json', data_files='../data/val_1.jsonl', split='train')

from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

# def formatting_func(example):
#     text = f"### Question: {example['text']}\n ### Answer: {example['labels_t']}"
#     return text

def formatting_func(example):
    prompt=f"""Identify the high-level disease category to which the following abstract from a medical text belongs by selecting from the options provided:

### Disease Categories:
1. Digestive System Diseases
2. Cardiovascular Diseases
3. Neoplasms
4. Nervous System Diseases
5. General Pathological Conditions


### Abstract:
{example['text']}

Answer: {example['labels_t']}"""
    return prompt

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

def generate_and_tokenize_prompt(prompt):
    return tokenizer(formatting_func(prompt))

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

import matplotlib.pyplot as plt

def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
    print(len(lengths))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.show()

# plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)

max_length = 800 # Setting max length to 800 basis data distribution

def generate_and_tokenize_prompt2(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt2)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt2)

# plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)

eval_prompt=f"""Identify the high-level disease category to which the following abstract from a medical text belongs by selecting from the options provided:

### Disease Categories:
1. Digestive System Diseases
2. Cardiovascular Diseases
3. Neoplasms
4. Nervous System Diseases
5. General Pathological Conditions


### Abstract:
Congenital myasthenia associated with facial malformations in Iraqi and Iranian Jews. A new genetic syndrome. Fourteen Jewish patients from 10 families of either Iraqi or Iranian origin with congenital myasthenia had associated facial malformations which included an elongated face, mandibular prognathism with class III malocclusion and a high-arched palate. Other common features were muscle weakness restricted predominantly to ptosis, weakness of facial and masticatory muscles, and fatigable speech; mild and nonprogressive course; response to cholinesterase inhibitors; absence of antibodies to acetylcholine receptor; decremental response on repetitive stimulation at 3 Hz but no repetitive compound muscle action potential in response to a single nerve stimulus. This newly recognized form of congenital myasthenia with distinctive ethnic clustering and associated facial malformations is transmitted as an autosomal recessive disorder. The facial abnormalities may be secondary to the neuromuscular defect or may be primary and unrelated. Further studies are needed to elucidate the defect in neuromuscular transmission responsible for the pathogenesis of this syndrome.

Answer:"""

# Init an eval tokenizer that doesn't add padding or eos token
eval_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
)

model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
print("Model evaluation before training--->")
with torch.no_grad():
    print(eval_tokenizer.decode(model.generate(**model_input, max_new_tokens=256, repetition_penalty=1.15)[0], skip_special_tokens=True))

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# print(model)

if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True

model = accelerator.prepare_model(model)

import transformers
from datetime import datetime

project = "disease-finetune"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "../" + run_name

training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_steps=3000,
        learning_rate=2.5e-5, # Want a small lr for finetuning
        fp16=True,
        optim="paged_adamw_8bit",
        logging_steps=300,              # When to start reporting loss
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=300,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=300,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        # report_to="wandb",           # Comment this out if you don't want to use weights & baises
        # run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    )

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

import pandas as pd
trainer_history_df=pd.DataFrame(trainer.state.log_history)

trainer_history_df.to_csv("../data/trainer_history.csv")