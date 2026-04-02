FROM python:3.10-slim

WORKDIR /app

# System dependencies just in case the datasets lib complains about missing compilers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Entrypoint — run Gradio app for HF Spaces
CMD ["python", "app.py"]
