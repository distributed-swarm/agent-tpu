FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install EdgeTPU native runtime (libedgetpu.so.1) from Coral APT repo.
# Debian slim does NOT include this package by default.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
 && mkdir -p /usr/share/keyrings \
 && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/coral-edgetpu.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    > /etc/apt/sources.list.d/coral-edgetpu.list \
 && apt-get update && apt-get install -y --no-install-recommends \
    libedgetpu1-std \
    usbutils \
 && rm -rf /var/lib/apt/lists/*

# NOTE: Coral wheels (pycoral + tflite-runtime 2.5.0.post1) live on the Coral pip index, not PyPI.
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --prefer-binary \
      --extra-index-url https://google-coral.github.io/py-repo/ \
      -r requirements.txt

COPY app.py worker_sizing.py ops_loader.py .
COPY ops ./ops

ENV CONTROLLER_URL="http://controller:8080"
ENV AGENT_NAME="agent-tpu-base"

CMD ["python", "app.py"]

