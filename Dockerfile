# Z-Image Turbo (+ LoRA) — RunPod serverless worker (ComfyUI based)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget ca-certificates python3 python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# PyTorch (CUDA 12.4 wheels)
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# ComfyUI (pin a ref for reproducible builds; override with --build-arg COMFYUI_REF=...)
ARG COMFYUI_REF=master
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /ComfyUI && \
    cd /ComfyUI && git checkout ${COMFYUI_REF} && \
    pip install -r requirements.txt

# Worker runtime deps
RUN pip install runpod websocket-client

# Worker files
COPY handler.py    /handler.py
COPY workflow.json /workflow.json
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Model + io dirs (diffusion_models / text_encoders / vae / loras get symlinked
# to the network volume at runtime by entrypoint.sh)
RUN mkdir -p /ComfyUI/models/diffusion_models \
             /ComfyUI/models/text_encoders \
             /ComfyUI/models/vae \
             /ComfyUI/models/loras \
             /ComfyUI/input \
             /ComfyUI/output

ENTRYPOINT ["/entrypoint.sh"]
