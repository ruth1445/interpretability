import argparse
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


def plot_heatmaps(tensor, tokens, neuron_ids, title, save_path, cmap):
    batch, seq_len, _ = tensor.shape
    for idx in range(min(batch, 4)):
        slice_ = tensor[idx, :, neuron_ids]
        plt.figure(figsize=(max(6, len(neuron_ids) / 6), max(3, seq_len / 3)))
        plt.imshow(slice_, aspect="auto", interpolation="nearest", cmap=cmap)
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
    x_len = len(tokens[0])
    plt.figure(figsize=(max(6, x_len / 2), 3.5))
    plt.plot(magnitudes, marker="o")
    plt.xticks(range(x_len), tokens[0], rotation=90)
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


def parse_args(cfg):
    parser = argparse.ArgumentParser(description="Visualize arithmetic conceptual circuit activations.")
    parser.add_argument("--lang", choices=cfg["languages"], default="en", help="Language to visualize.")
    parser.add_argument(
        "--kind",
        choices=["resid", "mlp_pre", "mlp_post"],
        default="mlp_post",
        help="Activation kind to visualize.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=cfg["layers_to_probe"][-1],
        help="Layer index to visualize (default: last configured layer).",
    )
    parser.add_argument("--topk", type=int, default=48, help="Number of neurons to display in heatmaps.")
    parser.add_argument("--cmap", default="magma", help="Matplotlib colormap to use for heatmaps.")
    parser.add_argument(
        "--heatmaps-only",
        action="store_true",
        help="If set, only the heatmap figures are generated (skip histogram and token magnitude plots).",
    )
    return parser.parse_args()


def main():
    cfg = load_cfg()
    args = parse_args(cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_dir = os.path.join(cfg["save_dir"], "activations", args.lang)
    figures_dir = os.path.join(cfg["save_dir"], "figures", f"{args.lang}_L{args.layer}")
    os.makedirs(figures_dir, exist_ok=True)

    tensor = get_layer_tensor(load_obj_array(os.path.join(base_dir, f"{args.kind}.npy")), args.layer)

    with open(os.path.join(cfg["prompts_dir"], f"{args.lang}.txt"), "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    token_text = token_strings(tokenizer, prompts[: tensor.shape[0]], max_len=tensor.shape[1])
    top_neurons = topk_neurons_by_variance(tensor, k=args.topk)

    title = f"{args.lang.upper()} {args.kind} layer {args.layer}"
    heatmap_root = os.path.join(figures_dir, f"heatmap_{args.kind}_L{args.layer}.png")
    plot_heatmaps(tensor, token_text, top_neurons, title, heatmap_root, cmap=args.cmap)

    if not args.heatmaps_only:
        plot_token_magnitude(
            tensor,
            token_text,
            f"{args.lang.upper()} mean |act| per token — {args.kind} L{args.layer}",
            os.path.join(figures_dir, f"token_magnitude_{args.kind}_L{args.layer}.png"),
        )
        plot_histogram(
            tensor,
            f"{args.lang.upper()} distribution — {args.kind} L{args.layer}",
            os.path.join(figures_dir, f"hist_{args.kind}_L{args.layer}.png"),
        )

    print(f"Saved figures to: {figures_dir}")


if __name__ == "__main__":
    main()
