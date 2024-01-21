
from model import Model
import numpy as np
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
 
disease_categories = {
    1: "digestive system diseases",
    2: "cardiovascular diseases",
    3: "neoplasms",
    4: "nervous system diseases",
    5: "general pathological conditions"
}
  
class Classifier:
    def __init__(self) -> None:
        self.model = Model.load_model()
        self.tokenizer = Model.load_tokenizer()
        pass
    def formatting_func(self,text):
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
    def infer_model(self,text:str):
        prompt = self.formatting_func(text)

        model_input = self.tokenizer(prompt, return_tensors="pt").to(device)
        generated_text = self.tokenizer.decode(self.model.generate(**model_input, max_new_tokens=10)[0], skip_special_tokens=True)

        return generated_text[generated_text.find('Answer:')+8:]


    def predict(self,text:str):
        pred_category=self.infer_model(text)
        return pred_category.strip(),None,0.7