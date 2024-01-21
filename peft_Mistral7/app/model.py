import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
from peft import PeftModel

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
base_model_id = "mistralai/Mistral-7B-v0.1"

class Model:
  
  """A model class to load the model and tokenizer"""

  def __init__(self) -> None:
    pass
  
  def load_model():

    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,  # Mistral, same as before
        quantization_config=bnb_config,  # Same quantization config as before
        device_map="auto",
        trust_remote_code=True)
    
    ft_model = PeftModel.from_pretrained(base_model, "./model_adapters/checkpoint-1500")
    ft_model.eval()

    return ft_model.to(device)


  def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)
    return tokenizer
