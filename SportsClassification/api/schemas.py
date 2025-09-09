from pydantic import BaseModel
from typing import Dict

class PredictResponse(BaseModel):
    predictions: Dict[str, float]
    top_k: int = 0  # optional with default

class LoadModelRequest(BaseModel):
    path: str