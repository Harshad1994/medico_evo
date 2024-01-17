import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
from peft import PeftModel


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
)

eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)



ft_model = PeftModel.from_pretrained(base_model, "./model_adapters/checkpoint-1500")
ft_model.eval()



def formatting_func(text):
    prompt=f"""Identify the high-level disease category to which the following abstract from a medical text belongs by selecting from the options provided:

### Disease Categories:
1. Digestive System Diseases
2. Cardiovascular Diseases
3. Neoplasms
4. Nervous System Diseases
5. General Pathological Conditions


### Abstract:
{text}

Answer:"""
    return prompt




@torch.no_grad()
def infer_model(text):

    prompt = formatting_func(text)

    model_input = eval_tokenizer(prompt, return_tensors="pt").to(device)
    generated_text = eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=10)[0], skip_special_tokens=True)

    return generated_text[generated_text.find('Answer:')+8:]



def predict(text):
    pred_category=infer_model(text)

    return pred_category,None,0.7

