import os, yaml, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

def load_cfg():
    with open("experiments/conceptual_circuits/configs.yaml","r") as f:
        return yaml.safe_load(f)

def prep_io(model_name, precision="bf16", device="auto"):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if precision=="bf16" and torch.cuda.is_available() else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    device_map=device,
    offload_folder="experiments/conceptual_circuits/offload" )
    mdl.eval()
    return tok, mdl

def read_prompts(prompts_dir, lang):
    with open(os.path.join(prompts_dir, f"{lang}.txt"), "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def register_hooks(model, layers_to_probe):
    cache = {"mlp_pre":[],"mlp_post":[],"resid":[]}
    handles = []

    # model-agnostic-ish: grab transformer blocks
    blocks = [m for m in model.modules() if m.__class__.__name__.lower().endswith("block") or m.__class__.__name__.lower().endswith("layer")]
    # fallback: many HF models expose model.layers or model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h

    def hook_resid(i):
        def fn(module, inp, out):
            cache["resid"].append((i, out.detach().float().cpu()))
        return fn

    def safe_detach(x):
    if isinstance(x, tuple):
        x = x[0]
    if hasattr(x, "detach"):
        return x.detach().float().cpu()
    else:
        return torch.tensor(x, dtype=torch.float32).cpu()

    def hook_mlp_pre(i):
        def fn(module, inp, out):
            cache["mlp_pre"].append((i, safe_detach(inp)))
        return fn

    def hook_mlp_post(i):
        def fn(module, inp, out):
        cache["mlp_post"].append((i, safe_detach(out)))
        return fn


    for i in layers_to_probe:
        # heuristic: each block usually has .mlp / .feed_forward
        blk = blocks[i]
        if hasattr(blk, "mlp"):
            handles += [blk.register_forward_hook(hook_resid(i)),
                        blk.mlp.register_forward_hook(hook_mlp_pre(i)),
                        blk.mlp.register_forward_hook(hook_mlp_post(i))]
        elif hasattr(blk, "feed_forward"):
            handles += [blk.register_forward_hook(hook_resid(i)),
                        blk.feed_forward.register_forward_hook(hook_mlp_pre(i)),
                        blk.feed_forward.register_forward_hook(hook_mlp_post(i))]
        else:
            handles += [blk.register_forward_hook(hook_resid(i))]

    return cache, handles

@torch.no_grad()
def run_lang(cfg, tok, mdl, lang):
    prompts = read_prompts(cfg["prompts_dir"], lang)
    os.makedirs(os.path.join(cfg["save_dir"], "activations", lang), exist_ok=True)

    cache, handles = register_hooks(mdl, cfg["layers_to_probe"])
    inputs = tok(prompts, return_tensors="pt", padding=True).to(mdl.device)
    out = mdl(**inputs, output_hidden_states=True)
    logits = out.logits[:, -1, :].float().cpu().numpy()

    # save logits + cache
    np.save(os.path.join(cfg["save_dir"], "activations", lang, "logits.npy"), logits)
    for k in cache:
        # each entry: list of (layer_idx, tensor[B, T, H])
        packed = [(i, t.numpy()) for (i, t) in cache[k]]
        np.save(os.path.join(cfg["save_dir"], "activations", lang, f"{k}.npy"), np.array(packed, dtype=object))

    for h in handles: h.remove()

def main():
    cfg = load_cfg(); set_seed(cfg["seed"])
    tok, mdl = prep_io(cfg["model_name"], cfg["precision"], cfg["device"])
    for lang in cfg["languages"]:
        run_lang(cfg, tok, mdl, lang)
    print("Saved activations & logits.")

if __name__ == "__main__":
    main()


