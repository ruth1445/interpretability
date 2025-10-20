import os
import yaml
import numpy as np
from transformers import AutoTokenizer


def load_cfg():
    with open("experiments/conceptual_circuits/arith/configs.yaml", "r") as f:
        return yaml.safe_load(f)


def load_logits(path):
    logits = np.load(path)
    # convert to probabilities for easier interpretation
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def format_topk(probs, tokenizer, k=5):
    idx = np.argsort(-probs)[:k]
    return [(tokenizer.convert_ids_to_tokens(int(i)), float(probs[i])) for i in idx]


def main():
    cfg = load_cfg()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for lang in cfg["languages"]:
        logits_path = os.path.join(cfg["save_dir"], "activations", lang, "logits.npy")
        if not os.path.exists(logits_path):
            print(f"[warn] No logits found for {lang} at {logits_path}")
            continue

        probs = load_logits(logits_path)
        prompts_file = os.path.join(cfg["prompts_dir"], f"{lang}.txt")
        with open(prompts_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

        print(f"\n=== {lang.upper()} ({len(prompts)} prompts) ===")
        for row_idx, (prompt, row_probs) in enumerate(zip(prompts, probs)):
            print(f"\nPrompt {row_idx}: {prompt}")
            top = format_topk(row_probs, tokenizer)
            print(" Top predictions:")
            for token, value in top:
                print(f"  - {token}: {value:.4f}")

            for target in cfg["token_targets"].get(lang, []):
                vocab = tokenizer.get_vocab()
                if target in vocab:
                    token_id = vocab[target]
                else:
                    token_id = tokenizer(target, add_special_tokens=False)["input_ids"][0]
                print(f"  Target {target}: {row_probs[token_id]:.4f}")


if __name__ == "__main__":
    main()
