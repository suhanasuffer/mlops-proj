# Use a lightweight Python base image
FROM python:3.10-slim-bookworm

WORKDIR /app

# Install only required system packages 
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip + install deps quickly with pinned versions
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir dvc[s3] awscli streamlit pytest \
    && rm -rf ~/.cache/pip

# Copy source code AFTER dependencies (so code changes don’t trigger full rebuild)
COPY . .

# RUN dvc repro || true

EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
