FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir tmm botorch gpytorch

COPY backend/ backend/
COPY frontend/ frontend/
COPY train_parametric_pinn.py .
COPY run_distill_optimize.py .
COPY start.py .

# 모델 체크포인트
COPY parametric_pinn_ckpt.pt ./
COPY fno_surrogate_ckpt.pt ./
COPY botorch_top5.pt ./
COPY pinn_results_all.pt ./
COPY fingerprint_sample.png ./

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
