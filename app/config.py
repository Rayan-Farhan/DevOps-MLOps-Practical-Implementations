import os
from typing import List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

class Settings:
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_PREFIX: str = os.getenv("API_PREFIX", "/api")
    
    CORS_ORIGINS: List[str] = [
        origin.strip() 
        for origin in os.getenv(
            "CORS_ORIGINS", 
            "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173"
        ).split(",")
        if origin.strip()
    ]
    
    # Allow all origins in non-production (for development convenience)
    CORS_ALLOW_ALL: bool = os.getenv("CORS_ALLOW_ALL", "true" if ENVIRONMENT != "production" else "false").lower() == "true"
    
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "app/logs/app.log")
    LOG_DIR: str = os.getenv("LOG_DIR", "app/logs")
    
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH", 
        str(Path(__file__).parent / "model" / "diabetes_model.pkl")
    )
    
    APP_NAME: str = os.getenv("APP_NAME", "Diabetes Prediction API")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    APP_DESCRIPTION: str = os.getenv(
        "APP_DESCRIPTION", 
        "Predict the likelihood of diabetes based on patient health parameters."
    )
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"
    
    def get_cors_origins(self) -> List[str]:
        # Return wildcard for development, specific origins for production
        if self.CORS_ALLOW_ALL and not self.is_production:
            return ["*"]
        return self.CORS_ORIGINS
    
    def __repr__(self) -> str:
        return f"Settings(environment={self.ENVIRONMENT}, api_port={self.API_PORT})"


settings = Settings()

