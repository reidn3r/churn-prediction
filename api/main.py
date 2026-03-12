from fastapi import FastAPI
from api.config.lifespan import lifespan, ml_models
from api.models.InferenceEndpointModel import InferenceInput

app = FastAPI(
  title="Churn Inferente API",
  lifespan=lifespan
)

@app.get('/ping')
def ping() -> str:
  return 'pong'


@app.post('/predict')
def predict(data: InferenceInput):
  model = ml_models["churn_v1"]
  proba = model.predict_proba(data.to_dataframe())
  return {
    "churn_probability": float(proba[0][1]),
    "churn": bool(proba[0][1] >= ml_models["threshold"])
  }