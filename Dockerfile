# Use NVIDIA PyTorch image
# FROM nvcr.io/nvidia/pytorch:23.11-py3
FROM python:3.10.13-slim-bullseye

# Set working directory
WORKDIR /workspace

COPY requirements.txt .

# Install Python dependencies for upgrading and Diffusion requirements.
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    python3 -m pip install --no-cache-dir --pre --upgrade --extra-index-url https://pypi.nvidia.com tensorrt && \
    rm -rf /root/.cache/pip/* && \
    apt-get purge -y --auto-remove

# Copy the application code
COPY . .

# Run the FastAPI server on container startup
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
