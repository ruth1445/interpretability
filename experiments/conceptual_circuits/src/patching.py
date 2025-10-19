import os, yaml, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_cfg():
    with open("experiments/conceptual_circuits/configs.yaml","r") as f: return yaml.safe_load(f)

@torch.no_grad()
def last_token_logit(model, tok, prompt, device):
    ids = tok(prompt, return_tensors="pt").to(device)
    out = model(**ids)
    return out.logits[:, -1, :].softmax(-1).cpu().numpy()[0]

def patch_layer_resid(model, layer_idx, src_hidden, hook_token_idx=-1):
    blocks = model.model.layers if hasattr(model.model, "layers") else model.transformer.h
    def fn(module, inp, out):
        replacement = torch.from_numpy(src_hidden)
        if isinstance(out, tuple):
            hidden = out[0]
            replacement = replacement.to(hidden.device)
            hidden = hidden.clone()
            hidden[..., hook_token_idx, :] = replacement
            return (hidden,) + out[1:]
        replacement = replacement.to(out.device)
        out = out.clone()
        out[..., hook_token_idx, :] = replacement
        return out
    return blocks[layer_idx].register_forward_hook(fn)

def main():
    cfg = load_cfg()
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    offload_dir = os.environ.get("HF_OFFLOAD_DIR", "/tmp/cc_offload")
    os.makedirs(offload_dir, exist_ok=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        device_map=cfg.get("device", "auto"),
        offload_folder=offload_dir
    )
    layer = cfg["layers_to_probe"][-1]

    # load a cached EN activation to patch into other languages
    # (take mean over batch/time for simplicity)
    en_resid = np.load(os.path.join(cfg["save_dir"], "activations", "en", "resid.npy"), allow_pickle=True)
    en_layer = [t for (i,t) in en_resid if int(i)==layer][0].mean(axis=(0,1))  # [H]

    prompts = {
      "de": "Das Gegenteil von klein ist",
      "ar": "عكس كلمة صغير هو",
      "ml": "ചെറുതിന്റെ വിരുദ്ധം"
    }
    targets = {"de":["groß"], "ar":["كبير"], "ml":["വലിയ"]}

    for lang, prompt in prompts.items():
        base = last_token_logit(mdl, tok, prompt, mdl.device)
        handle = patch_layer_resid(mdl, layer, en_layer)
        patched = last_token_logit(mdl, tok, prompt, mdl.device)
        handle.remove()

        for t in targets[lang]:
            tid = tok.convert_tokens_to_ids(t) if t in tok.get_vocab() else tok(t, add_special_tokens=False)["input_ids"][0]
            print(lang, t, "Δprob =", float(patched[tid] - base[tid]))

if __name__ == "__main__":
    main()
