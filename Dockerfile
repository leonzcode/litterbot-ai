FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenCV + ultralytics need a few shared libs and a TTF font for labels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU-only torch from the CPU index, then everything else from PyPI.
# The CPU wheel is ~200MB; the default CUDA wheel is ~2GB+ and useless on a
# CPU-only host.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install -r requirements.txt

COPY app.py .
COPY .streamlit ./.streamlit
COPY models ./models
COPY yolov8n.pt .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
