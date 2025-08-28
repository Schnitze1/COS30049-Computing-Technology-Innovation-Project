import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils.model_io import list_models, load_model
from config import get_model_dir
from utils.predict import run_prediction


class PredictRequest(BaseModel):
    model: str
    instances: List[List[float]]
    # Example for Swagger UI
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "random_forest",
                    "instances": [
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                    ]
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    model: str
    predictions: List[int]
    probabilities: Optional[List[float]] = None


app = FastAPI(title="Model Inference API", version="1.0.0")

# CORS for local dev frontends (React/Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


API_PREFIX = "/api/v1"


@app.get(f"{API_PREFIX}/health")
def health():
    return {"status": "ok"}


@app.get(f"{API_PREFIX}/models")
def get_models():
    return list_models(out_dir=get_model_dir())


@app.post(f"{API_PREFIX}/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    models = list_models(out_dir=get_model_dir())
    if req.model not in models:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found")
    model = load_model(req.model, out_dir=get_model_dir())
    preds, proba = run_prediction(model, req.instances)
    return PredictResponse(model=req.model, predictions=preds, probabilities=proba)


@app.post(f"{API_PREFIX}/models/{{model_name}}/predict", response_model=PredictResponse)
def predict_by_path(model_name: str, instances: List[List[float]]):
    models = list_models(out_dir=get_model_dir())
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    model = load_model(model_name, out_dir=get_model_dir())
    preds, proba = run_prediction(model, instances)
    return PredictResponse(model=model_name, predictions=preds, probabilities=proba)


if __name__ == "__main__":
    import uvicorn
    os.makedirs(get_model_dir(), exist_ok=True)
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)


