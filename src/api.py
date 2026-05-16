import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from tokenizers import Tokenizer
from scipy.special import softmax
from onnxruntime import InferenceSession
from contextlib import asynccontextmanager


class ModelSingleton:
    _instance = None
    classifier = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()

            cls.tokenizer = Tokenizer.from_file(
                "models/tokenizer/tokenizer.json")

            cls.session = InferenceSession("models/distilbert/model.onnx")
            cls.id2label = {0: "NEGATIVE", 1: "POSITIVE"}

        return cls._instance


class TextIn(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    ModelSingleton.get()  # load on startup
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
def predict(body: TextIn) -> dict:

    encoded = ModelSingleton.get().tokenizer.encode(body.text)
    inputs = {
        "input_ids": np.array([encoded.ids]),
        "attention_mask": np.array([encoded.attention_mask]),
    }
    logits = ModelSingleton.get().session.run(None, inputs)[0]
    probs = softmax(logits[0])
    predicted_id = np.argmax(probs)
    return {
        "label": ModelSingleton.get().id2label[predicted_id],
        "score": float(probs[predicted_id])
    }


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}
