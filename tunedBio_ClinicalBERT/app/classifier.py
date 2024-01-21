
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

    @torch.no_grad()
    def predict(self, text:str):

        inputs = self.tokenizer([text], max_length=512,padding="max_length", truncation=True, return_tensors = 'pt')
        inputs = inputs.to(device)
        logits = self.model(**inputs).logits.detach()

        
        logits_soft = logits.softmax(dim=1)[0]
        index = logits_soft.argmax().item()
        confidence = logits_soft[index].item()

        prediction = disease_categories[index+1]

        return prediction, index+1,confidence