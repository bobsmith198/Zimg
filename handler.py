"""
RunPod serverless worker — Z-Image Turbo (text-to-image) with LoRA + optional img2img.

Pattern (same as the rapid-qwen worker): boot ComfyUI, patch a workflow.json, queue it
over the ComfyUI API/websocket, return the result as a base64 image.

The big difference from a numeric-ID handler: this patches nodes BY TITLE (the
"_meta.title" field). So you can export your own working Z-Image workflow from ComfyUI
(Save (API Format)), give the key nodes the titles listed below, drop it in as
workflow.json, and this handler keeps working — no code edits.

Required node titles in workflow.json:
  "Load Diffusion Model"  (UNETLoader)
  "Load Text Encoder"     (CLIPLoader)
  "Load VAE"              (VAELoader)
  "Positive Prompt"       (CLIPTextEncode)
  "Negative Prompt"       (CLIPTextEncode)
  "Empty Latent"          (EmptySD3LatentImage)
  "Sampler"               (KSampler)
  "VAE Decode"            (VAEDecode)
  "Save Image"            (SaveImage)
Optional:
  "LoRA Loader"           (LoraLoaderModelOnly)  -- required only if you pass `loras`

Input (job["input"]):
  prompt            (str, required)
  negative_prompt   (str, default "")
  loras             (list[{path, scale}] or single object) -- path = https URL (HF/Civitai
                       direct link) or a filename already on the volume's loras dir;
                       scale = LoRA strength (default 1.0). Multiple are chained.
  width, height     (int)  OR  size ("1024*1024" / "1024x1024")   default 1024x1024
  steps             (int, default 8)        -- Z-Image Turbo: ~6-10
  cfg               (float, default 1.0)    -- distilled model; ~1.0 (negative prompt weak)
  sampler           (str, default "euler")
  scheduler         (str, default "simple")
  seed              (int, default random; -1 also = random)
  denoise           (float)  -- t2i default 1.0; img2img default 0.5 (also reads `strength`)
  image / image_url / image_base64 / image_path  (optional) -> enables img2img
  output_format     ("png"|"jpeg", default "png")
  ckpt_name / clip_name / vae_name / clip_type  (optional) -> override model files/encoder type

Output: { "image": "<base64>", "seed": <int> }   or   { "error": "..." }
"""
import runpod
import os
import json
import uuid
import time
import copy
import base64
import shutil
import random
import logging
import urllib.request
import subprocess

import websocket  # provided by the websocket-client package

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("zimage-worker")

SERVER   = os.getenv("SERVER_ADDRESS", "127.0.0.1")
COMFY    = f"http://{SERVER}:8188"
CLIENT   = str(uuid.uuid4())
WS_URL   = f"ws://{SERVER}:8188/ws?clientId={CLIENT}"
IN_DIR   = "/ComfyUI/input"
OUT_DIRS = ["/ComfyUI/output", "/ComfyUI/temp"]
LORA_DIR = "/ComfyUI/models/loras"
WORKFLOW = os.getenv("WORKFLOW_PATH", "/workflow.json")


# ── ComfyUI lifecycle ───────────────────────────────────────────
def wait_for_comfyui(timeout=600):
    log.info("Waiting for ComfyUI…")
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{COMFY}/", timeout=3)
            log.info("ComfyUI ready")
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError("ComfyUI did not start in time")


# ── small helpers ───────────────────────────────────────────────
def to_16(v):
    return max(16, int(round(float(v) / 16.0) * 16))


def parse_size(inp):
    if inp.get("size"):
        s = str(inp["size"]).lower().replace("x", "*")
        try:
            w, h = s.split("*")[:2]
            return to_16(w), to_16(h)
        except Exception:
            pass
    return to_16(inp.get("width", 1024)), to_16(inp.get("height", 1024))


def save_base64(b64_data, out_path):
    clean = b64_data.split(",", 1)[1] if "," in b64_data else b64_data
    clean = "".join(clean.split())
    clean = clean.rstrip("=")
    clean += "=" * ((4 - len(clean) % 4) % 4)
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(clean))
    return out_path


def download(url, out_path, timeout=600):
    r = subprocess.run(
        ["wget", "-O", out_path, "--no-verbose",
         "--user-agent", "Mozilla/5.0 (compatible)", url],
        capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(f"download failed for {url}: {r.stderr[-400:]}")
    return out_path


def resolve_image(inp):
    """Optional source image for img2img. Returns the filename inside COMFY_INPUT, or None."""
    src = None
    if inp.get("image_path"):
        os.makedirs(IN_DIR, exist_ok=True)
        name = f"{uuid.uuid4().hex[:8]}_src.png"
        dst = os.path.join(IN_DIR, name)
        shutil.copy(inp["image_path"], dst)
        return name
    src = inp.get("image") or inp.get("image_url") or inp.get("image_base64")
    if not src:
        return None
    os.makedirs(IN_DIR, exist_ok=True)
    name = f"{uuid.uuid4().hex[:8]}_src.png"
    dst = os.path.join(IN_DIR, name)
    if isinstance(src, str) and src.startswith("http"):
        download(src, dst)
    else:
        save_base64(src, dst)
    return name


# ── workflow graph patching (by node title) ─────────────────────
def title_index(wf):
    idx = {}
    for nid, node in wf.items():
        t = (node.get("_meta", {}) or {}).get("title")
        if t and t not in idx:
            idx[t] = nid
    return idx


def nid_for(idx, title):
    if title not in idx:
        raise RuntimeError(f"workflow.json is missing a node titled '{title}'")
    return idx[title]


def set_in(wf, nid, **kw):
    wf[nid].setdefault("inputs", {}).update(kw)


def consumers(wf, src_id, slot=0):
    """Every (node_id, input_key) whose value is the link [src_id, slot]."""
    hits = []
    for nid, node in wf.items():
        for k, v in (node.get("inputs", {}) or {}).items():
            if isinstance(v, list) and len(v) == 2 \
               and str(v[0]) == str(src_id) and int(v[1]) == slot:
                hits.append((nid, k))
    return hits


def bypass(wf, nid, passthrough_key="model", slot=0):
    """Repoint everything reading this node's output to the node's `passthrough_key`
    source, then delete the node. Used to drop the LoRA loader when no LoRA is set."""
    src = wf[nid]["inputs"][passthrough_key]
    for cid, ckey in consumers(wf, nid, slot):
        wf[cid]["inputs"][ckey] = src
    wf.pop(nid, None)


# ── ComfyUI API ─────────────────────────────────────────────────
def queue_prompt(prompt):
    data = json.dumps({"prompt": prompt, "client_id": CLIENT}).encode()
    req = urllib.request.Request(f"{COMFY}/prompt", data=data)
    res = json.loads(urllib.request.urlopen(req).read())
    if "error" in res:
        raise RuntimeError(
            f"ComfyUI rejected the workflow: {res['error']} | "
            f"node_errors={res.get('node_errors', {})}"
        )
    return res["prompt_id"]


def get_history(pid):
    with urllib.request.urlopen(f"{COMFY}/history/{pid}") as r:
        return json.loads(r.read())


def run_workflow(ws, prompt):
    pid = queue_prompt(prompt)
    log.info(f"queued {pid}")
    while True:
        msg = ws.recv()
        if isinstance(msg, str):
            d = json.loads(msg)
            if d.get("type") == "executing":
                data = d.get("data", {})
                if data.get("node") is None and data.get("prompt_id") == pid:
                    break
            elif d.get("type") == "execution_error":
                raise RuntimeError(f"ComfyUI execution error: {d}")
    outputs = get_history(pid)[pid].get("outputs", {})
    for _, nout in outputs.items():
        for img in nout.get("images", []):
            for base in OUT_DIRS:
                p = os.path.join(base, img.get("subfolder", ""), img["filename"])
                if os.path.exists(p):
                    log.info(f"found image {p}")
                    with open(p, "rb") as f:
                        return base64.b64encode(f.read()).decode("utf-8")
    return None


# ── handler ─────────────────────────────────────────────────────
def handler(job):
    inp = job.get("input", {}) or {}
    if not inp.get("prompt"):
        return {"error": "'prompt' is required"}

    wf = copy.deepcopy(json.load(open(WORKFLOW)))
    idx = title_index(wf)

    w, h    = parse_size(inp)
    steps   = int(inp.get("steps", 8))
    cfg     = float(inp.get("cfg", 1.0))
    sampler = inp.get("sampler", "euler")
    sched   = inp.get("scheduler", "simple")
    seed    = int(inp.get("seed", -1))
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    # Prompts + latent size
    set_in(wf, nid_for(idx, "Positive Prompt"), text=inp["prompt"])
    set_in(wf, nid_for(idx, "Negative Prompt"), text=inp.get("negative_prompt", ""))
    set_in(wf, nid_for(idx, "Empty Latent"), width=w, height=h, batch_size=1)

    # Always patch model filenames and CLIP type so workflow.json defaults can
    # never cause a validation mismatch.  Env vars mirror entrypoint.sh; job
    # input takes priority over env vars, which take priority over literals.
    set_in(wf, nid_for(idx, "Load Diffusion Model"),
           unet_name=inp.get("ckpt_name") or os.getenv("ZIMAGE_DIFFUSION_FILE", "perfeczion_10BF16.safetensors"))
    set_in(wf, nid_for(idx, "Load VAE"),
           vae_name=inp.get("vae_name")   or os.getenv("ZIMAGE_VAE_FILE",       "ae.safetensors"))
    set_in(wf, nid_for(idx, "Load Text Encoder"),
           clip_name=inp.get("clip_name") or os.getenv("ZIMAGE_TEXTENC_FILE",   "qwen_3_4b.safetensors"),
           type=inp.get("clip_type")      or os.getenv("ZIMAGE_CLIP_TYPE",      "lumina2"))

    # ── LoRA(s) ──────────────────────────────────────────────────
    lora_node = idx.get("LoRA Loader")  # optional node
    loras = inp.get("loras") or []
    if isinstance(loras, dict):
        loras = [loras]

    def lora_filename(spec):
        path = spec.get("path") or spec.get("name") or ""
        if path.startswith("http"):
            os.makedirs(LORA_DIR, exist_ok=True)
            fn = os.path.basename(path.split("?")[0]) or f"{uuid.uuid4().hex}.safetensors"
            dest = os.path.join(LORA_DIR, fn)
            if not os.path.exists(dest):
                download(path, dest)
            return fn
        return path  # assume already present in models/loras

    if loras:
        if not lora_node:
            return {"error": "workflow.json has no node titled 'LoRA Loader' but `loras` was provided"}
        first = loras[0]
        set_in(wf, lora_node,
               lora_name=lora_filename(first),
               strength_model=float(first.get("scale", 1.0)))
        prev = lora_node
        for i, spec in enumerate(loras[1:], start=1):
            new_id = f"lora_extra_{i}"
            wf[new_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": [prev, 0],
                    "lora_name": lora_filename(spec),
                    "strength_model": float(spec.get("scale", 1.0)),
                },
                "_meta": {"title": f"LoRA Loader {i + 1}"},
            }
            prev = new_id
        if prev != lora_node:  # repoint the sampler to the end of the chain
            set_in(wf, nid_for(idx, "Sampler"), model=[prev, 0])
    elif lora_node:
        bypass(wf, lora_node, passthrough_key="model")  # no LoRA -> drop the loader

    # ── optional img2img ─────────────────────────────────────────
    src = resolve_image(inp)
    # denoise: honor `denoise`, then `strength` (their existing client sends `strength`),
    # else sensible default. Anything outside (0,1] falls back.
    denoise = inp.get("denoise", inp.get("strength", None))
    try:
        denoise = float(denoise)
        if denoise <= 0 or denoise > 1:
            raise ValueError
    except (TypeError, ValueError):
        denoise = 0.5 if src else 1.0

    if src:
        vae = nid_for(idx, "Load VAE")
        wf["src_load"] = {
            "class_type": "LoadImage",
            "inputs": {"image": src},
            "_meta": {"title": "Source Image"},
        }
        wf["src_encode"] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["src_load", 0], "vae": [vae, 0]},
            "_meta": {"title": "Encode Source"},
        }
        set_in(wf, nid_for(idx, "Sampler"), latent_image=["src_encode", 0])

    set_in(wf, nid_for(idx, "Sampler"),
           seed=seed, steps=steps, cfg=cfg,
           sampler_name=sampler, scheduler=sched, denoise=denoise)

    # ── run ──────────────────────────────────────────────────────
    ws = websocket.WebSocket()
    for attempt in range(10):
        try:
            ws.connect(WS_URL)
            break
        except Exception as e:
            log.warning(f"WS connect failed ({attempt + 1}/10): {e}")
            if attempt == 9:
                raise
            time.sleep(3)
    try:
        image_b64 = run_workflow(ws, wf)
    finally:
        ws.close()
        if src:
            fp = os.path.join(IN_DIR, src)
            if os.path.exists(fp):
                os.remove(fp)

    if not image_b64:
        return {"error": "no image output found"}
    return {"image": image_b64, "seed": seed}


wait_for_comfyui()
runpod.serverless.start({"handler": handler})