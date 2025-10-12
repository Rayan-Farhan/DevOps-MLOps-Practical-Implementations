from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
import logging
import os

app = FastAPI(
    title="Diabetes Prediction API",
    description="Predict the likelihood of diabetes based on patient health parameters.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later to frontend URL (e.g., http://localhost:5173)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

os.makedirs("app/logs", exist_ok=True)

logging.basicConfig(
    filename="app/logs/app.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

@app.on_event("startup")
def startup_event():
    logging.info("FastAPI app started successfully!")

@app.on_event("shutdown")
def shutdown_event():
    logging.info("FastAPI app shutting down.")

@app.get("/")
def root():
    return {"message": "Welcome to the Diabetes Prediction API!"}