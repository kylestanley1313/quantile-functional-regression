import copy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim


__all__ = [
    'train_flow',
    'flow_encode',
    'flow_decode',
    'save_flow',
    'load_flow'
]


# ----- Define scale/shift nets -----

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ----- RealNVP coupling layer -----

class RealNVPLayer(nn.Module):
    def __init__(self, dim, mask, hidden=64):
        super().__init__()
        self.mask = mask
        self.s_net = MLP(dim, dim, hidden=hidden)
        self.t_net = MLP(dim, dim, hidden=hidden)

    def forward(self, x):
        x_masked = x * self.mask
        s = torch.tanh(self.s_net(x_masked)) * (1 - self.mask)  # bounded
        t = self.t_net(x_masked) * (1 - self.mask)

        y = x_masked + (1 - self.mask) * ((x - t) * torch.exp(-s))
        logdet = -s.sum(dim=1)
        return y, logdet

    def inverse(self, y):
        y_masked = y * self.mask
        s = torch.tanh(self.s_net(y_masked)) * (1 - self.mask)
        t = self.t_net(y_masked) * (1 - self.mask)

        x = y_masked + (1 - self.mask) * (y * torch.exp(s) + t)
        return x


# ----- Full flow -----

class NormalizingFlow(nn.Module):
    def __init__(self, dim, n_layers=16, hidden=64):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):  # TODO: Why alternate between even and odd partitions?
            mask = torch.tensor([i % 2] * (dim // 2) + [(i+1) % 2] * (dim - dim // 2),
                                dtype=torch.float32)
            self.layers.append(RealNVPLayer(dim, mask, hidden=hidden))

    def forward(self, x):
        logdet_sum = torch.zeros(x.size(0))
        for layer in self.layers:
            x, logdet = layer(x)
            logdet_sum += logdet
        return x, logdet_sum

    def inverse(self, z):
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z


# ----- Training -----

def train_flow(C, n_layers, hidden=64, max_epochs=200, lr=1e-3, weight_decay=0.0,
               val_frac=0.15, patience=20, min_delta=1e-4, seed=12345, print_epoch=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dim = C.shape[1]
    flow = NormalizingFlow(dim, n_layers=n_layers, hidden=hidden)
    optimizer = optim.Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)

    base = torch.distributions.MultivariateNormal(
        torch.zeros(dim),
        torch.eye(dim)
    )
    C = torch.tensor(C, dtype=torch.float32)

    n = C.shape[0]
    C_val = None
    if val_frac and val_frac > 0:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        n_val = round(val_frac * n)
        if n_val >= 1 and n - n_val >= 1:
            val_idx, train_idx = perm[:n_val], perm[n_val:]
            C_train, C_val = C[train_idx], C[val_idx]
        else:
            C_train = C
    else:
        C_train = C

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        z, logdet = flow(C_train)
        log_prob = base.log_prob(z) + logdet
        loss = -log_prob.mean()
        loss.backward()
        optimizer.step()

        val_loss = None
        if C_val is not None:
            with torch.no_grad():
                z_val, logdet_val = flow(C_val)
                val_loss = -(base.log_prob(z_val) + logdet_val).mean().item()

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state = copy.deepcopy(flow.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        if print_epoch is not None and ((epoch + 1) % print_epoch == 0 or (epoch + 1) == max_epochs):
            if val_loss is not None:
                print(f"Epoch {epoch + 1}, loss={loss.item():.4f}, val_loss={val_loss:.4f}")
            else:
                print(f"Epoch {epoch + 1}, loss={loss.item():.4f}")

        if C_val is not None and epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1} (no val improvement for {patience} epochs)")
            break

    if best_state is not None:
        flow.load_state_dict(best_state)

    return flow


# ----- Flow Encode/Decode -----

@torch.no_grad()
def flow_encode(C, flow):
    Z, _ = flow(torch.tensor(C, dtype=torch.float32))
    return Z.numpy()

@torch.no_grad()
def flow_decode(Z, flow):
    C = flow.inverse(torch.tensor(Z, dtype=torch.float32))
    return C.numpy()


# ----- Flow File Management -----

def save_flow(flow, path):
    torch.save(flow.state_dict(), path)

def load_flow(dim, n_layers, path, hidden=64):
    flow = NormalizingFlow(dim, n_layers=n_layers, hidden=hidden)
    flow.load_state_dict(torch.load(path, map_location="cpu"))
    flow.eval()
    return flow
