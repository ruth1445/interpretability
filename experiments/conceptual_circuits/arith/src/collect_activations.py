import os
import yaml
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def load_cfg():
    with open("experiments/conceptual_circuits/arith/configs.yaml", "r") as f:
        return yaml.safe_load(f)


def resolve_offload_dir():
    offload_dir = os.environ.get("HF_OFFLOAD_DIR", "experiments/conceptual_circuits/offload")
    os.makedirs(offload_dir, exist_ok=True)
    return offload_dir


def prep_io(model_name, precision="bf16", device="auto"):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if precision == "bf16" and torch.cuda.is_available() else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        offload_folder=resolve_offload_dir(),
    )
    mdl.eval()
    return tok, mdl


def read_prompts(prompts_dir, lang):
    with open(os.path.join(prompts_dir, f"{lang}.txt"), "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def safe_detach_all(x):
    if isinstance(x, (tuple, list)):
        out = []
        for item in x:
            out.extend(safe_detach_all(item))
        return out
    if hasattr(x, "detach"):
        return [x.detach().float().cpu()]
    return []


def register_hooks(model, layers_to_probe):
    cache = {"mlp_pre": [], "mlp_post": [], "resid": []}
    handles = []

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        blocks = model.model.decoder.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
    else:
        raise ValueError("Unable to locate transformer blocks on this model.")

    def hook_resid(layer_idx):
        def fn(module, _inp, out):
            for t in safe_detach_all(out):
                cache["resid"].append((layer_idx, t))
        return fn

    def hook_mlp_pre(layer_idx):
        def fn(module, inp, _out):
            for t in safe_detach_all(inp):
                cache["mlp_pre"].append((layer_idx, t))
        return fn

    def hook_mlp_post(layer_idx):
        def fn(module, _inp, out):
            for t in safe_detach_all(out):
                cache["mlp_post"].append((layer_idx, t))
        return fn

    for idx in layers_to_probe:
        block = blocks[idx]
        handles.append(block.register_forward_hook(hook_resid(idx)))
        if hasattr(block, "mlp"):
            handles.append(block.mlp.register_forward_hook(hook_mlp_pre(idx)))
            handles.append(block.mlp.register_forward_hook(hook_mlp_post(idx)))
        elif hasattr(block, "feed_forward"):
            handles.append(block.feed_forward.register_forward_hook(hook_mlp_pre(idx)))
            handles.append(block.feed_forward.register_forward_hook(hook_mlp_post(idx)))
        elif hasattr(block, "fc1") and hasattr(block, "fc2"):
            handles.append(block.fc1.register_forward_hook(hook_mlp_pre(idx)))
            handles.append(block.fc2.register_forward_hook(hook_mlp_post(idx)))
        else:
            # fallback: only residual stream captured
            pass

    return cache, handles


@torch.no_grad()
def run_lang(cfg, tok, mdl, lang):
    print(f"\n▶ Processing language: {lang}")
    prompts = read_prompts(cfg["prompts_dir"], lang)
    lang_dir = os.path.join(cfg["save_dir"], "activations", lang)
    os.makedirs(lang_dir, exist_ok=True)

    cache, handles = register_hooks(mdl, cfg["layers_to_probe"])
    inputs = tok(prompts, return_tensors="pt", padding=True).to(mdl.device)
    outputs = mdl(**inputs, output_hidden_states=True)
    logits = outputs.logits[:, -1, :].float().cpu().numpy()
    np.save(os.path.join(lang_dir, "logits.npy"), logits)

    for key, values in cache.items():
        packed = [(int(i), t.numpy()) for (i, t) in values]
        np.save(os.path.join(lang_dir, f"{key}.npy"), np.array(packed, dtype=object))

    for handle in handles:
        handle.remove()

    print(f"✅ Saved activations for {lang}")


def main():
    cfg = load_cfg()
    set_seed(cfg["seed"])
    tok, mdl = prep_io(cfg["model_name"], cfg["precision"], cfg["device"])
    for lang in cfg["languages"]:
        run_lang(cfg, tok, mdl, lang)
    print("\n🎯 Arithmetic activations captured!")


if __name__ == "__main__":
    main()
