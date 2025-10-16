import os, yaml, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# -------------------- CONFIG LOADER --------------------
def load_cfg():
    with open("experiments/conceptual_circuits/configs.yaml", "r") as f:
        return yaml.safe_load(f)

# -------------------- MODEL LOADER --------------------
def prep_io(model_name, precision="bf16", device="auto"):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if precision == "bf16" and torch.cuda.is_available() else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        offload_folder="experiments/conceptual_circuits/offload"
    )
    mdl.eval()
    return tok, mdl

# -------------------- PROMPT READER --------------------
def read_prompts(prompts_dir, lang):
    with open(os.path.join(prompts_dir, f"{lang}.txt"), "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

# -------------------- SAFE DETACH (universal) --------------------
def safe_detach_all(x):
    """
    Handles tensors or nested tuples/lists of tensors safely.
    Returns a list of detached CPU tensors.
    """
    if isinstance(x, (tuple, list)):
        out = []
        for item in x:
            out.extend(safe_detach_all(item))
        return out
    elif hasattr(x, "detach"):
        return [x.detach().float().cpu()]
    else:
        return []

# -------------------- HOOK REGISTRATION --------------------
def register_hooks(model, layers_to_probe):
    cache = {"mlp_pre": [], "mlp_post": [], "resid": []}
    handles = []

    # heuristic: most models define blocks here
    blocks = [m for m in model.modules() if m.__class__.__name__.lower().endswith(("block", "layer"))]

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h

    def hook_resid(i):
        def fn(module, inp, out):
            for t in safe_detach_all(out):
                cache["resid"].append((i, t))
        return fn

    def hook_mlp_pre(i):
        def fn(module, inp, out):
            for t in safe_detach_all(inp):
                cache["mlp_pre"].append((i, t))
        return fn

    def hook_mlp_post(i):
        def fn(module, inp, out):
            for t in safe_detach_all(out):
                cache["mlp_post"].append((i, t))
        return fn

    for i in layers_to_probe:
        blk = blocks[i]
        if hasattr(blk, "mlp"):
            handles += [
                blk.register_forward_hook(hook_resid(i)),
                blk.mlp.register_forward_hook(hook_mlp_pre(i)),
                blk.mlp.register_forward_hook(hook_mlp_post(i)),
            ]
        elif hasattr(blk, "feed_forward"):
            handles += [
                blk.register_forward_hook(hook_resid(i)),
                blk.feed_forward.register_forward_hook(hook_mlp_pre(i)),
                blk.feed_forward.register_forward_hook(hook_mlp_post(i)),
            ]
        else:
            handles += [blk.register_forward_hook(hook_resid(i))]

    return cache, handles

# -------------------- RUN PER LANGUAGE --------------------
@torch.no_grad()
def run_lang(cfg, tok, mdl, lang):
    print(f"\n▶ Processing language: {lang}")
    prompts = read_prompts(cfg["prompts_dir"], lang)
    os.makedirs(os.path.join(cfg["save_dir"], "activations", lang), exist_ok=True)

    cache, handles = register_hooks(mdl, cfg["layers_to_probe"])
    inputs = tok(prompts, return_tensors="pt", padding=True).to(mdl.device)
    out = mdl(**inputs, output_hidden_states=True)
    logits = out.logits[:, -1, :].float().cpu().numpy()

    np.save(os.path.join(cfg["save_dir"], "activations", lang, "logits.npy"), logits)

    for k in cache:
        packed = [(i, t.numpy()) for (i, t) in cache[k]]
        np.save(os.path.join(cfg["save_dir"], "activations", lang, f"{k}.npy"), np.array(packed, dtype=object))

    for h in handles:
        h.remove()

    print(f"✅ Saved activations for {lang}")

# -------------------- MAIN --------------------
def main():
    cfg = load_cfg()
    set_seed(cfg["seed"])
    tok, mdl = prep_io(cfg["model_name"], cfg["precision"], cfg["device"])
    for lang in cfg["languages"]:
        run_lang(cfg, tok, mdl, lang)
    print("\n🎯 All activations & logits saved successfully!")

if __name__ == "__main__":
    main()
