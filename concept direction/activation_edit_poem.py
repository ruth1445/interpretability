import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from typing import Dict, Optional
from collections import Counter

MODEL_NAME = "gpt2"
TARGET_LAYER = 8
MAX_NEW_TOKENS = 50
MIN_WORDS = 7
MAX_WORDS = 16
TEMPERATURE = 0.8
TOP_P = 0.88
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_ATTEMPTS = 40

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True
).to(DEVICE).eval()

PUNCT_STRIP = ".!,?;:\"'"


def enforce_length(words, last_word, include_word: Optional[str]):
    if len(words) <= MAX_WORDS:
        return words
    keep = words[-(MAX_WORDS - 1):]
    keep[-1] = last_word
    if include_word and include_word.lower() not in [w.lower() for w in keep]:
        keep.insert(max(len(keep) - 1, 0), include_word)
        if len(keep) > MAX_WORDS:
            keep = keep[-MAX_WORDS:]
            keep[-1] = last_word
    return keep


def ensure_constraints(line: str, last_word: str, include_word: Optional[str]) -> str:
    words = [w.strip(PUNCT_STRIP) for w in line.strip().replace("\n", " ").split() if w.strip(PUNCT_STRIP)]
    if include_word and include_word.lower() not in [w.lower() for w in words]:
        insert_at = max(len(words) - 1, 0)
        words.insert(insert_at, include_word)
    if words:
        words[-1] = last_word
    else:
        words = [include_word] if include_word else []
        words.append(last_word)
    words = enforce_length(words, last_word, include_word)
    return " ".join(words)


def looks_instructional(line: str) -> bool:
    lower = line.lower()
    banned_starts = ("write", "include", "don't", "do not", "please", "if you", "this gives")
    if any(lower.startswith(b) for b in banned_starts):
        return True
    banned_words = {"must", "should", "instruction", "example", "sentence"}
    return any(word in lower for word in banned_words)


def has_excessive_repetition(line: str) -> bool:
    words = [w.lower() for w in line.split()]
    if not words:
        return True
    counts = Counter(words)
    return counts.most_common(1)[0][1] > max(3, len(words) // 2)


@torch.no_grad()
def gather_hidden_states(sequence_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    outputs = model(
        input_ids=sequence_ids.to(DEVICE),
        output_hidden_states=True,
        return_dict=True
    )
    layer_hidden = outputs.hidden_states[TARGET_LAYER][0]
    return layer_hidden[prompt_len:].cpu()


@torch.no_grad()
def generate_line_with_activations(
    prompt: str,
    last_word: str,
    include_word: Optional[str] = None,
    max_new_tokens: int = MAX_NEW_TOKENS
) -> Dict[str, torch.Tensor]:
    example_line = f"Silver rain hums low across the sleepy field {last_word}."
    base_prompt = (
        f"{prompt.strip()} Here is an example poetic line ending with '{last_word}':\n"
        f"\"{example_line}\"\n"
        "Now compose a fresh poetic line in a similarly vivid style."
    )
    if include_word:
        base_prompt += f" Slip the word '{include_word}' somewhere before the ending."
    base_prompt += " Reply only with the new line.\nPoetic line:"

    prompt_inputs = tok(base_prompt, return_tensors="pt")
    prompt_ids = prompt_inputs["input_ids"][0]
    prompt_len = prompt_ids.shape[-1]

    chosen_line = None
    attempts_used = 0

    for attempts_used in range(1, MAX_ATTEMPTS + 1):
        generation = model.generate(
            **{k: v.to(DEVICE) for k, v in prompt_inputs.items()},
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            pad_token_id=tok.eos_token_id,
            return_dict_in_generate=True
        )
        generated_ids = generation.sequences[0][prompt_len:]
        generated_text = tok.decode(generated_ids, skip_special_tokens=True)
        line_candidate = generated_text.split("\n")[0].strip()
        if looks_instructional(line_candidate):
            continue
        constrained_line = ensure_constraints(line_candidate, last_word, include_word)
        if len(constrained_line.split()) < MIN_WORDS:
            continue
        if has_excessive_repetition(constrained_line):
            continue
        chosen_line = constrained_line
        break

    if chosen_line is None:
        raise RuntimeError(f"Failed to craft a satisfactory line after {MAX_ATTEMPTS} attempts.")

    line_ids = tok(
        chosen_line,
        return_tensors="pt",
        add_special_tokens=False
    )["input_ids"][0]

    full_sequence = torch.cat([prompt_ids, line_ids], dim=0).unsqueeze(0)
    activations = gather_hidden_states(full_sequence, prompt_len)
    tokens = tok.convert_ids_to_tokens(line_ids)

    return {
        "line": chosen_line,
        "tokens": tokens,
        "activations": activations,
        "attempts": attempts_used,
    }


if __name__ == "__main__":
    prompts = [
        ("Write an evocative harvest image in verse.", "carrot", None),
        ("Write an energetic woodland image in verse.", "rabbit", None),
        ("Write an evocative harvest image in verse with lush greens.", "carrot", "green"),
        ("Write an energetic woodland image in verse with lush greens.", "rabbit", "green"),
    ]

    results = []
    for idx, (prompt, last_word, include_word) in enumerate(prompts, start=1):
        result = generate_line_with_activations(prompt, last_word, include_word)
        results.append(result)
        print(f"Line {idx}: {result['line']} (attempt {result['attempts']})")

    save_path = Path(__file__).parent / "captured_activations.pt"
    torch.save({f"line_{i+1}": r["activations"] for i, r in enumerate(results)}, save_path)
    print(f"\nSaved activations to {save_path}")
