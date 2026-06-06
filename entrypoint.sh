#!/usr/bin/env bash
set -euo pipefail

# ── Model storage ────────────────────────────────────────────────
# Keep the big weights on the RunPod network volume so only the FIRST cold start
# pays the download; later workers just symlink. Falls back to container disk if
# no volume is mounted (not recommended — bloats the image / re-downloads).
VOLUME="${MODEL_VOLUME:-/runpod-volume}"
[ -d "$VOLUME" ] || VOLUME="/ComfyUI/models"
MODELS="$VOLUME/zimage"
mkdir -p "$MODELS/diffusion_models" "$MODELS/text_encoders" "$MODELS/vae" "$MODELS/loras"

# Model sources — override any of these via env vars without rebuilding the image.
DIFF_URL="${ZIMAGE_DIFFUSION_URL:-https://civitai.red/api/download/models/2836778?fileId=2723038}"
CLIP_URL="${ZIMAGE_TEXTENC_URL:-https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors}"
VAE_URL="${ZIMAGE_VAE_URL:-https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors}"

DIFF_FILE="${ZIMAGE_DIFFUSION_FILE:-perfeczion_10BF16.safetensors}"
CLIP_FILE="${ZIMAGE_TEXTENC_FILE:-qwen_3_4b.safetensors}"
VAE_FILE="${ZIMAGE_VAE_FILE:-ae.safetensors}"

fetch () {  # $1=url  $2=dest
  if [ -f "$2" ]; then
    echo "cached: $(basename "$2")"
  else
    echo "downloading $(basename "$2") …"
    # Civitai gated downloads: set CIVITAI_TOKEN to send an auth header.
    if echo "$1" | grep -q "civitai" && [ -n "${CIVITAI_TOKEN:-}" ]; then
      wget --no-verbose --header "Authorization: Bearer ${CIVITAI_TOKEN}" \
           --user-agent "Mozilla/5.0 (compatible)" -O "$2" "$1"
    # HF gated/large files: set HF_TOKEN to send an auth header.
    elif [ -n "${HF_TOKEN:-}" ]; then
      wget --no-verbose --header "Authorization: Bearer ${HF_TOKEN}" \
           --user-agent "Mozilla/5.0 (compatible)" -O "$2" "$1"
    else
      wget --no-verbose --user-agent "Mozilla/5.0 (compatible)" -O "$2" "$1"
    fi
  fi
}

fetch "$DIFF_URL" "$MODELS/diffusion_models/$DIFF_FILE"
fetch "$CLIP_URL" "$MODELS/text_encoders/$CLIP_FILE"
fetch "$VAE_URL"  "$MODELS/vae/$VAE_FILE"

# ── Symlink volume dirs into ComfyUI ─────────────────────────────
link_dir () {  # $1=src  $2=dst
  rm -rf "$2"
  ln -s "$1" "$2"
}
link_dir "$MODELS/diffusion_models" /ComfyUI/models/diffusion_models
link_dir "$MODELS/text_encoders"    /ComfyUI/models/text_encoders
link_dir "$MODELS/vae"              /ComfyUI/models/vae
link_dir "$MODELS/loras"            /ComfyUI/models/loras

# ── Boot ComfyUI, then the serverless handler ────────────────────
echo "starting ComfyUI…"
python -u /ComfyUI/main.py --listen 127.0.0.1 --port 8188 ${COMFY_ARGS:-} &

exec python -u /handler.py