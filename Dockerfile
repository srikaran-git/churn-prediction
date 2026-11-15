# Base image
FROM python:3.11-slim

# Working directory
WORKDIR /app

# Dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/pipeline_v1.pkl ./models/pipeline_v1.pkl

# Port
EXPOSE 8000

# Start command
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
