import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


#Finetuned model is uploaded on huggingface hub. This model will be used here in this application.
model_name = 'HarshadKunjir/BioBERT_medical_abstract_classification'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Model:
  """A model class to load the model and tokenizer"""

  def __init__(self) -> None:
    pass
  
  def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model

  def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer