import os, yaml, numpy as np, matplotlib.pyplot as plt
from transformers import AutoTokenizer

# ---------------- cfg helpers ----------------
def load_cfg():
    with open("experiments/conceptual_circuits/configs.yaml","r") as f:
        return yaml.safe_load(f)

def load_obj_array(path):
    arr = np.load(path, allow_pickle=True)
    return [(int(i), t) for (i, t) in arr]

def get_layer_tensor(obj_array, layer_idx):
    hits = [t for (i,t) in obj_array if i == layer_idx]
    if not hits: raise ValueError(f"No entries for layer {layer_idx}")
    # If multiple hooks fired per layer, concatenate on batch
    X = np.concatenate(hits, axis=0)  # [B,T,H]
    return X

# --------------- core viz --------------------
def topk_neurons_by_var(X, k=64):
    # X: [B,T,H] -> select top-k neurons by variance across B*T
    B,T,H = X.shape
    flat = X.reshape(B*T, H)
    vari = flat.var(axis=0)
    idx = np.argsort(-vari)[:k]
    return idx

def token_strings(tok, prompts, max_len=None):
    # tokenize and decode tokens for each prompt
    toks = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
    ids = toks["input_ids"].tolist()
    out = []
    for row in ids:
        s = [tok.convert_ids_to_tokens(i) for i in row]
        if max_len: s = s[:max_len]
        out.append(s)
    return out

def heatmap_per_prompt(X, token_text, neuron_idx, title, save_path):
    # X: [B,T,H], token_text: list[list[str]], neuron_idx: [K]
    B,T,H = X.shape
    K = len(neuron_idx)
    for b in range(min(B, 4)):  # show up to 4 prompts
        mat = X[b, :, neuron_idx]  # [T,K]
        plt.figure(figsize=(max(6, K/6), max(3, T/3)))
        plt.imshow(mat, aspect='auto', interpolation='nearest')
        plt.colorbar()
        xt = [str(i) for i in neuron_idx]
        plt.xticks(range(K), xt, rotation=90)
        tok_row = token_text[b]
        plt.yticks(range(len(tok_row)), tok_row)
        plt.title(f"{title} — prompt {b}")
        plt.tight_layout()
        sp = save_path.replace(".png", f"_p{b}.png")
        plt.savefig(sp, dpi=160)
        plt.close()

def line_mean_abs_per_token(X, token_text, title, save_path):
    # plot mean |activation| per token to see where the layer focuses
    B,T,H = X.shape
    mag = np.mean(np.abs(X), axis=(0,2))  # [T] averaged over batch & neurons
    plt.figure(figsize=(max(6, T/2), 3.5))
    plt.plot(mag, marker='o')
    labels = token_text[0]
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def hist_distribution(X, title, save_path):
    plt.figure(figsize=(5.5,3.5))
    plt.hist(X.flatten(), bins=120)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

# --------------- entrypoint -------------------
def main():
    cfg = load_cfg()
    model_name = cfg["model_name"]
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # choose what to visualize
    lang = "ar"                 # change: en/de/ar/ml
    layer = cfg["layers_to_probe"][-1]  # or 5/10, etc.
    kind = "mlp_post"           # one of: resid, mlp_pre, mlp_post

    save_dir = os.path.join(cfg["save_dir"], "figures", f"{lang}_L{layer}")
    os.makedirs(save_dir, exist_ok=True)

    # load activations
    base = os.path.join(cfg["save_dir"], "activations", lang)
    arr = load_obj_array(os.path.join(base, f"{kind}.npy"))
    X = get_layer_tensor(arr, layer)     # [B,T,H]

    # reconstruct the exact prompts you used (first N lines of the prompt file)
    with open(os.path.join(cfg["prompts_dir"], f"{lang}.txt"), "r", encoding="utf-8") as f:
        prompts = [l.strip() for l in f if l.strip()]
    # ensure token_text matches shapes; pad/truncate to X.shape[1]
    tok_text = token_strings(tok, prompts[:X.shape[0]], max_len=X.shape[1])

    # choose top-K most variant neurons to show in heatmaps
    idx = topk_neurons_by_var(X, k=64)

    # 1) Heatmaps (token × top-K neurons) for up to 4 prompts
    heatmap_per_prompt(
        X, tok_text, idx,
        title=f"{lang.upper()} | {kind} | layer {layer}",
        save_path=os.path.join(save_dir, f"heatmap_{kind}_L{layer}.png")
    )

    # 2) Mean |activation| per token (which positions the layer cares about)
    line_mean_abs_per_token(
        X, tok_text,
        title=f"{lang.upper()} | mean |act| per token | {kind} L{layer}",
        save_path=os.path.join(save_dir, f"token_magnitude_{kind}_L{layer}.png")
    )

    # 3) Distribution histogram (are we polarizing more in deeper layer?)
    hist_distribution(
        X,
        title=f"{lang.upper()} | distribution {kind} L{layer}",
        save_path=os.path.join(save_dir, f"hist_{kind}_L{layer}.png")
    )

    print(f"Saved figures to: {save_dir}")

if __name__ == "__main__":
    main()
