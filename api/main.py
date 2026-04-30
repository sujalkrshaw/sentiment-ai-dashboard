from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

class InputText(BaseModel):
    text: str

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(data: InputText):
    pred = model.predict([data.text])[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([data.text])[0]
        confidence = max(proba)
    else:
        confidence = 0.5

    return {"prediction": int(pred), "confidence": float(confidence)}