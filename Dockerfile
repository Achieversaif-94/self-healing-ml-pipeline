# ============================================
# FastAPI Backend Dockerfile
# ============================================
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (needed for some ML libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY 05_fastapi_app.py .
COPY retrain.py .
COPY create_new_data.py .

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "05_fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]