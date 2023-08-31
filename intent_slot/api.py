import json
from jsonschema import validate
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from model import SequenceClassifierModel
from config import API_CONFIG#, API_SCHEMA


class APIRequest(BaseModel):
    text: str

model = SequenceClassifierModel(
    API_CONFIG["base_model"],
    API_CONFIG["num_labels"],
    API_CONFIG["learning_rate"],
)

app = FastAPI()

@app.post('/predict')
def predict(request: APIRequest):
    """
    Test with:
        curl -X POST http://127.0.0.1:5000/predict
            -H 'Content-Type: application/json'
            -d '{"text": "This is a test."}'
    """
    result = model.predict(text_list=[request.text])

    return JSONResponse(
        content = {
            "text": request.text,
            "prediction": result["prediction"],
            "scores": result["scores"]
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        #debug=True,
    )
