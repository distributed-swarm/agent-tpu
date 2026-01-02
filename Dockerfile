FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --- System deps + Coral Edge TPU runtime repo ---
# usbutils -> lsusb
# libedgetpu + edgetpu-examples -> runtime + known-good models/examples
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg \
      usbutils \
    && rm -rf /var/lib/apt/lists/*

# Add Coral Debian repo (for libedgetpu / edgetpu-examples)
RUN set -eux; \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor -o /usr/share/keyrings/coral.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/coral.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
      > /etc/apt/sources.list.d/coral-edgetpu.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      libedgetpu1-std \
      edgetpu-examples; \
    rm -rf /var/lib/apt/lists/*

# --- Python deps ---
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# --- App ---
COPY app.py worker_sizing.py ops_loader.py .
COPY ops ./ops

ENV CONTROLLER_URL="http://controller:8080"
ENV AGENT_NAME="agent-docker-1"

CMD ["python", "app.py"]
