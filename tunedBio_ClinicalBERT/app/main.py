import uvicorn
from fastapi import FastAPI
import uuid
from pydantic import BaseModel
from backend import predict
from classifier import Classifier


class InputRequest(BaseModel):
    abstract: str

class ResponseModel(BaseModel):
    res_id: str
    category: str
    category_index: int | None
    confidence: float

app = FastAPI()
classifier = Classifier()

@app.post("/classify")
def classify(request_json: InputRequest) -> ResponseModel | dict:
 
    print("hitting the endpoint")

    res_id=str(uuid.uuid4())

    text = request_json.abstract

    prediction,cat_index,prob = classifier.predict(text)

    response_dict = {"res_id":res_id,"category":prediction,'category_index':cat_index,"confidence":prob}

    return ResponseModel(**response_dict)


if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
