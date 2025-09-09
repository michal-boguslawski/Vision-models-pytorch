import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch as T
from api.schemas import PredictResponse, LoadModelRequest
from inference.infer import InferenceRunner
from typing import Optional


device = "cuda" if T.cuda.is_available() else "cpu"
runner: Optional[InferenceRunner] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global runner
    path = "logs/sports_cv_project_classification/vgg11-paper-pretrained-w-augmenations/config.yaml"
    app.state.runner = await asyncio.to_thread(InferenceRunner, config_path=path, device=device)
    yield
    # Clean up the ML models and release the resources
    runner = None


app = FastAPI(title="Sports Classification API", lifespan=lifespan)


@app.get("/health")
async def health():
    if not hasattr(app.state, "runner") or app.state.runner is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/load_model")
async def load_model(request: LoadModelRequest):
    try:
        app.state.runner = await asyncio.to_thread(InferenceRunner, config_path=request.path, device=device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "Model loaded"}


@app.post("/predict/", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), top_k: int = 0):
    if not hasattr(app.state, "runner") or app.state.runner is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Run your inference (mocked here)
    image = Image.open(file.file).convert("RGB")
    result = await asyncio.to_thread(app.state.runner.predict, image=image)
    
    # Return as Pydantic model
    return PredictResponse(predictions=result, top_k=top_k)
