import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


model_name = 'HarshadKunjir/BioBERT_medical_abstract_classification'

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# quantized_model = torch.quantization.quantize_dynamic(
#     model, {torch.nn.Linear}, dtype=torch.qint8
# )
model.eval()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)



disease_categories = {
    1: "digestive system diseases",
    2: "cardiovascular diseases",
    3: "neoplasms",
    4: "nervous system diseases",
    5: "general pathological conditions"
}

@torch.no_grad()
def predict(text):

    inputs = tokenizer([text], max_length=512,padding="max_length", truncation=True, return_tensors = 'pt')
    inputs = inputs.to(device)
    logits = model(**inputs).logits.detach()

    
    logits_soft = logits.softmax(dim=1)[0]
    index = logits_soft.argmax().item()
    confidence = logits_soft[index].item()

    prediction = disease_categories[index+1]

    return prediction, index+1,confidence
    
