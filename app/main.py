from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.config import settings
import logging
import os
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION
)

allowed_origins = settings.get_cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.API_PREFIX)

os.makedirs(settings.LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=settings.LOG_FILE,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
)

@app.on_event("startup")
def startup_event():
    logging.info("FastAPI app started successfully!")
    # Expose default Prometheus metrics, including per-endpoint latency and count
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

@app.on_event("shutdown")
def shutdown_event():
    logging.info("FastAPI app shutting down.")

@app.get("/")
def root():
    return {"message": "Welcome to the Diabetes Prediction API!"}