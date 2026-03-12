from contextlib import asynccontextmanager
from fastapi import FastAPI
import mlflow

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
  mlflow.set_tracking_uri("http://localhost:5000")
  version = 1
  uri = f"models:/RandomForest_ChurnPredictor/{version}"
  
  ml_models[f"churn_v{version}"] = mlflow.sklearn.load_model(uri)
  ml_models["threshold"] = 0.374
  yield