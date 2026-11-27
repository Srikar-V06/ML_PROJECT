FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt streamlit fastapi uvicorn fairlearn

COPY src/ src/
COPY app/ app/
COPY config/ config/
COPY artifacts/p3/model_manifest.json artifacts/p3/model_manifest.json
COPY artifacts/p3/models/ artifacts/p3/models/
COPY artifacts/p3/stack/ artifacts/p3/stack/
COPY artifacts/p3/tables/ artifacts/p3/tables/
COPY artifacts/p4/calibration/ artifacts/p4/calibration/
COPY artifacts/p4/tables/ artifacts/p4/tables/
COPY artifacts/p4/plots/ artifacts/p4/plots/
COPY artifacts/p5/ artifacts/p5/
COPY artifacts/p1p2_300k/models/ artifacts/p1p2_300k/models/

EXPOSE 8501 8000
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
