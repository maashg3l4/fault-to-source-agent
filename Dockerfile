FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch to keep the image light
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install requirements
COPY requirements.txt .
# Add --default-timeout=1000 to prevent network timeouts
RUN pip install --no-cache-dir --default-timeout=1000 --retries 5 -r requirements.txt

# THIS IS THE MISSING LINE: Keeps the container alive
CMD ["tail", "-f", "/dev/null"]