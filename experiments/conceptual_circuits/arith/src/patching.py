import os
import yaml
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_cfg():
    with open("experiments/conceptual_circuits/arith/configs.yaml", "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def last_token_probs(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)
    return outputs.logits[:, -1, :].softmax(dim=-1).cpu().numpy()[0]


def patch_layer_resid(model, layer_idx, src_hidden, hook_token_idx=-1):
    blocks = model.model.layers if hasattr(model.model, "layers") else model.transformer.h

    def hook_fn(_module, _inp, out):
        replacement = torch.from_numpy(src_hidden)
        if isinstance(out, tuple):
            hidden = out[0].clone()
            replacement = replacement.to(hidden.device)
            hidden[..., hook_token_idx, :] = replacement
            return (hidden,) + out[1:]
        out = out.clone()
        replacement = replacement.to(out.device)
        out[..., hook_token_idx, :] = replacement
        return out

    return blocks[layer_idx].register_forward_hook(hook_fn)


def resolve_offload_dir():
    offload_dir = os.environ.get("HF_OFFLOAD_DIR", "experiments/conceptual_circuits/offload")
    os.makedirs(offload_dir, exist_ok=True)
    return offload_dir


def main():
    cfg = load_cfg()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        device_map=cfg.get("device", "auto"),
        offload_folder=resolve_offload_dir(),
    )
    layer = cfg["layers_to_probe"][-1]

    activations_path = os.path.join(cfg["save_dir"], "activations", "en", "resid.npy")
    english_resid = np.load(activations_path, allow_pickle=True)
    # average cached english activations across prompts/tokens
    en_layer_mean = [tensor for (idx, tensor) in english_resid if int(idx) == layer][0].mean(axis=(0, 1))

    prompts = {
        "en": "Seven plus five equals",
        "ar": "ما حاصل قسمة عشرين على خمسة هو",
        "de": "Die Aussage 'Zwei ist größer als Eins' ist",
    }
    targets = {
        "en": ["twelve"],
        "ar": ["أربعة"],
        "de": ["wahr"],
    }

    for lang, prompt in prompts.items():
        base = last_token_probs(model, tokenizer, prompt, model.device)
        handle = patch_layer_resid(model, layer, en_layer_mean)
        patched = last_token_probs(model, tokenizer, prompt, model.device)
        handle.remove()

        for token in targets[lang]:
            vocab = tokenizer.get_vocab()
            if token in vocab:
                idx = vocab[token]
            else:
                idx = tokenizer(token, add_special_tokens=False)["input_ids"][0]
            delta = float(patched[idx] - base[idx])
            print(f"{lang} {token} Δprob = {delta:.6f}")


if __name__ == "__main__":
    main()
