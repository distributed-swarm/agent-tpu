FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    usbutils \
 && rm -rf /var/lib/apt/lists/*

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
