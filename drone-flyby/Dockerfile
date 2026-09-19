FROM python:3.11-slim

# opencv needs libGL and libglib even in headless use.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# On linux/amd64 the default PyPI torch wheel is a CUDA build, so an NVIDIA GPU
# works as long as the container can see it: run with
#   docker run --gpus all -p 9053:9053 drone-flyby
# (needs the NVIDIA Container Toolkit on the host). Without --gpus the detector
# runs on the CPU and the server logs "!!! DETECTOR ON CPU" when it finds GPU
# hardware it cannot use. Check with: curl http://<host>:9053/api
RUN pip install --no-cache-dir -r requirements.txt

# Your model and the server. The sample data under src/ is not copied: it is
# 400 MB and only local_evaluator.py needs it.
COPY dtos.py utils.py example.py api.py ./
COPY pipeline/ ./pipeline/
# Train first (python train_detector.py) so weights/detector.pt exists.
COPY weights/ ./weights/

EXPOSE 9053

CMD ["python", "api.py"]
