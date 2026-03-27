FROM python:3.11-slim

RUN pip install --no-cache-dir "mlflow-skinny>=2.10.0"

EXPOSE 5000

# Use SQLite as the backend store (file:// is deprecated and broken on Windows paths)
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlruns/mlflow.db", \
     "--default-artifact-root", "/mlartifacts"]
