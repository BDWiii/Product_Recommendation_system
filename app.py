from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
from training.train import train
from recommendation.predict import recommend

app = FastAPI()

DATA_DIR = "api_temp"
os.makedirs(DATA_DIR, exist_ok=True)


class TrainRequest(BaseModel):
    users_file: str
    items_file: str
    interactions_file: str
    output_model: str = "LightFM_weights/model.pkl"


class PredictRequest(BaseModel):
    user_file: str
    items_file: str
    model_file: str = "LightFM_weights/model.pkl"
    top_k: int = 6


@app.post("/train/")
async def train_model(req: TrainRequest):
    train(
        req.users_file, req.items_file, req.interactions_file, output=req.output_model
    )
    return {"message": "Model trained successfully", "model_path": req.output_model}


@app.post("/predict/")
async def predict_items(req: PredictRequest):
    recs = recommend(req.model_file, req.user_file, req.items_file, top_k=req.top_k)
    return {"recommendations": recs}
