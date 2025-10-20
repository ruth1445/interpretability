import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


def load_cfg():
    with open("experiments/conceptual_circuits/arith/configs.yaml", "r") as f:
        return yaml.safe_load(f)


def load_obj_array(path):
    arr = np.load(path, allow_pickle=True)
    return [(int(idx), tensor) for (idx, tensor) in arr]


def get_layer_tensor(obj_array, layer_idx):
    tensors = [tensor for (idx, tensor) in obj_array if idx == layer_idx]
    if not tensors:
        raise ValueError(f"No cached activations for layer {layer_idx}")
    return np.concatenate(tensors, axis=0)


def topk_neurons_by_variance(tensor, k=48):
    batch, tokens, hidden = tensor.shape
    flat = tensor.reshape(batch * tokens, hidden)
    variances = flat.var(axis=0)
    return np.argsort(-variances)[:k]


def token_strings(tokenizer, prompts, max_len=None):
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
    rows = encoded["input_ids"].tolist()
    decoded = []
    for row in rows:
        toks = [tokenizer.convert_ids_to_tokens(idx) for idx in row]
        if max_len is not None:
            toks = toks[:max_len]
        decoded.append(toks)
    return decoded


def plot_heatmaps(tensor, tokens, neuron_ids, title, save_path):
    batch, seq_len, _ = tensor.shape
    for idx in range(min(batch, 4)):
        slice_ = tensor[idx, :, neuron_ids]
        plt.figure(figsize=(max(6, len(neuron_ids) / 6), max(3, seq_len / 3)))
        plt.imshow(slice_, aspect="auto", interpolation="nearest")
        plt.colorbar()
        plt.xticks(range(len(neuron_ids)), [str(i) for i in neuron_ids], rotation=90)
        plt.yticks(range(len(tokens[idx])), tokens[idx])
        plt.title(f"{title} — prompt {idx}")
        plt.tight_layout()
        target_path = save_path.replace(".png", f"_p{idx}.png")
        plt.savefig(target_path, dpi=160)
        plt.close()


def plot_token_magnitude(tensor, tokens, title, save_path):
    magnitudes = np.mean(np.abs(tensor), axis=(0, 2))
    plt.figure(figsize=(max(6, len(tokens[0]) / 2), 3.5))
    plt.plot(magnitudes, marker="o")
    plt.xticks(range(len(tokens[0])), tokens[0], rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_histogram(tensor, title, save_path):
    plt.figure(figsize=(5.5, 3.5))
    plt.hist(tensor.flatten(), bins=120)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def main():
    cfg = load_cfg()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lang = "en"
    layer = cfg["layers_to_probe"][-1]
    kind = "mlp_post"

    base_dir = os.path.join(cfg["save_dir"], "activations", lang)
    figures_dir = os.path.join(cfg["save_dir"], "figures", f"{lang}_L{layer}")
    os.makedirs(figures_dir, exist_ok=True)

    tensor = get_layer_tensor(load_obj_array(os.path.join(base_dir, f"{kind}.npy")), layer)

    with open(os.path.join(cfg["prompts_dir"], f"{lang}.txt"), "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    token_text = token_strings(tokenizer, prompts[:tensor.shape[0]], max_len=tensor.shape[1])
    top_neurons = topk_neurons_by_variance(tensor, k=48)

    title = f"{lang.upper()} {kind} layer {layer}"
    heatmap_root = os.path.join(figures_dir, f"heatmap_{kind}_L{layer}.png")
    plot_heatmaps(tensor, token_text, top_neurons, title, heatmap_root)

    plot_token_magnitude(
        tensor,
        token_text,
        f"{lang.upper()} mean |act| per token — {kind} L{layer}",
        os.path.join(figures_dir, f"token_magnitude_{kind}_L{layer}.png"),
    )
    plot_histogram(
        tensor,
        f"{lang.upper()} distribution — {kind} L{layer}",
        os.path.join(figures_dir, f"hist_{kind}_L{layer}.png"),
    )

    print(f"Saved figures to: {figures_dir}")


if __name__ == "__main__":
    main()
