from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import tensorflow as tf

app = FastAPI()

# Load model and tokenizer once at startup
model_path = "C:/Users/DELL/Desktop/newmodel/tf_model3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)

# Request body schema
class TranslationRequest(BaseModel):
    text: str
    max_length: int = 128

@app.post("/translate/")
def translate(req: TranslationRequest):
    inputs = tokenizer([req.text], return_tensors="tf")
    outputs = model.generate(**inputs, max_length=req.max_length)
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"translation": translated}
