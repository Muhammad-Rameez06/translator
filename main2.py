from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import tensorflow as tf
import easyocr
import cv2
import numpy as np
import shutil
import os

app = FastAPI()

# Load translation model and tokenizer
model_path = "C:/Users/DELL/Desktop/newmodel/tf_model4"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Text-to-text translation endpoint
class TranslationRequest(BaseModel):
    text: str
    max_length: int = 512

@app.post("/translate/")
def translate(req: TranslationRequest):
    inputs = tokenizer([req.text], return_tensors="tf")
    outputs = model.generate(**inputs, max_length=req.max_length)
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"translation": translated}

# OCR + translation endpoint
@app.post("/ocr-translate/")
async def ocr_translate(file: UploadFile = File(...), max_length: int = 512):
    # Save uploaded image temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Perform OCR
    result = reader.readtext(temp_path)
    os.remove(temp_path)

    # Join detected text without adding dots
    paragraph = ' '.join([item[1].strip() for item in result])

    # Translate the text
    inputs = tokenizer([paragraph], return_tensors="tf")
    outputs = model.generate(**inputs, max_length=max_length)
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "ocr_text": paragraph,
        "translation": translated
    }
