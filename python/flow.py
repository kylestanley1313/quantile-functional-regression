import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim


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
    def __init__(self, dim, mask):
        super().__init__()
        self.mask = mask
        self.s_net = MLP(dim, dim)
        self.t_net = MLP(dim, dim)

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
    def __init__(self, dim, n_layers=16):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask = torch.tensor([i % 2] * (dim // 2) + [(i+1) % 2] * (dim - dim // 2),
                                dtype=torch.float32)
            self.layers.append(RealNVPLayer(dim, mask))

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

def train_flow(C, n_layers, epochs=200, lr=1e-3, seed=12345):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dim = C.shape[1]
    flow = NormalizingFlow(dim, n_layers=n_layers)
    optimizer = optim.Adam(flow.parameters(), lr=lr)

    base = torch.distributions.MultivariateNormal(
        torch.zeros(dim),
        torch.eye(dim)
    )
    C = torch.tensor(C, dtype=torch.float32)

    for epoch in range(epochs):
        optimizer.zero_grad()
        z, logdet = flow(C)
        log_prob = base.log_prob(z) + logdet
        loss = -log_prob.mean()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, loss={loss.item():.4f}")

    return flow


# ----- Flow File Management -----

def save_flow(flow, path):
    torch.save(flow.state_dict(), path)

def load_flow(dim, n_layers, path):
    flow = NormalizingFlow(dim, n_layers=n_layers)
    flow.load_state_dict(torch.load(path, map_location="cpu"))
    flow.eval()
    return flow


# ----- Flow Encode/Decode -----

def flow_encode(c, flow):
    with torch.no_grad():
        z, _ = flow(torch.tensor(c, dtype=torch.float32))
    return z.numpy()

def flow_decode(z, flow):
    with torch.no_grad():
        c = flow.inverse(torch.tensor(z, dtype=torch.float32))
    return c.numpy()
