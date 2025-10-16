import os, numpy as np, torch, torch.nn as nn, torch.optim as optim, yaml

class SAE(nn.Module):
    def __init__(self, d_in, d_hidden=4096, l1=1e-3):
        super().__init__()
        self.enc = nn.Linear(d_in, d_hidden, bias=False)
        self.dec = nn.Linear(d_hidden, d_in, bias=False)
        self.l1 = l1
    def forward(self, x):
        z = torch.relu(self.enc(x))
        x_hat = self.dec(z)
        return x_hat, z

def load_cfg():
    with open("experiments/conceptual_circuits/configs.yaml","r") as f: return yaml.safe_load(f)

def load_layer_acts(save_dir, lang, key="mlp_post", layer_idx=10):
    arr = np.load(os.path.join(save_dir, "activations", lang, f"{key}.npy"), allow_pickle=True)
    # filter for layer
    tensors = [t for (i,t) in arr if int(i)==layer_idx]
    X = np.concatenate([t.reshape(-1, t.shape[-1]) for t in tensors], axis=0)
    return X

def train_sae(X, d_hidden=4096, l1=1e-3, epochs=3, bs=1024, lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu"):
    X = torch.from_numpy(X).to(device)
    d_in = X.shape[-1]
    model = SAE(d_in, d_hidden=d_hidden, l1=l1).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            batch = X[perm[i:i+bs]]
            xhat, z = model(batch)
            rec = ((batch - xhat)**2).mean()
            sparsity = z.abs().mean()
            loss = rec + model.l1*sparsity
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def save_sae(model, path): torch.save(model.state_dict(), path)

def main():
    cfg = load_cfg()
    # automatically detect valid layers
    arr = np.load(os.path.join(cfg["save_dir"], "activations", cfg["languages"][0], "mlp_post.npy"), allow_pickle=True)
    available_layers = sorted(set(int(i) for (i, _) in arr))
    layer = available_layers[-1]  # use last one by default
    print(f"Training SAE on layer {layer} (available layers: {available_layers})")

    X = []
    for lang in cfg["languages"]:
        Xi = load_layer_acts(cfg["save_dir"], lang, key="mlp_post", layer_idx=layer)
        if Xi.size > 0:
            X.append(Xi)
    if not X:
        raise ValueError("No activations found for the selected layer. Check your layer indices.")
    X = np.vstack(X)


if __name__ == "__main__":
    main()

