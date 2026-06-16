import torch


def ensure_2d_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor has shape (N, J).
    Accepts:
      - (J,)  -> (1, J)
      - (N,J) -> unchanged
    """
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        return tensor
    else:
        raise ValueError(f"tensor must be 1D or 2D, got shape {tensor.shape}")


def differentiate(Q, p):
    """
    Q: (B, J) or (J,)
    p: (J,)
    returns dQ/dp with same shape as Q
    central differences interior, one-sided boundaries
    """
    if Q.ndim == 1:
        Q = Q.unsqueeze(0)  # (1,J)

    B, J = Q.shape
    p = p.unsqueeze(0)  # (1,J)

    dQ = torch.zeros_like(Q)

    # interior: (Q[j+1] - Q[j-1]) / (p[j+1] - p[j-1])
    dQ[:,1:-1] = (Q[:,2:] - Q[:,:-2]) / (p[:,2:] - p[:,:-2])

    # boundaries one-sided
    dQ[:,0]  = (Q[:,1] - Q[:,0]) / (p[:,1] - p[:,0])
    dQ[:,-1] = (Q[:,-1] - Q[:,-2]) / (p[:,-1] - p[:,-2])

    return dQ if dQ.shape[0] > 1 else dQ.squeeze(0)


def integrate(dQ, p, p_star, Q_star):
    """
    dQ: (B,J) or (J,)
    p: (J,)
    p_star: scalar
    Q_star: scalar or (B,)
    returns Q: same shape as dQ
    """
    # --- Normalize shape ---
    single = False
    if dQ.ndim == 1:
        dQ = dQ.unsqueeze(0)  # (1,J)
        single = True

    B, J = dQ.shape
    p = p.unsqueeze(0)  # (1,J)

    # --- Resolve p_star to index ---
    with torch.no_grad():
        idx = torch.argmin(torch.abs(p - p_star)).item()

    # --- Resolve Q_star semantics ---
    if Q_star is None:
        raise ValueError("Q_star=None not supported here; handle in model.")
    
    if isinstance(Q_star, torch.Tensor):
        # Tensor cases
        if Q_star.ndim == 0:
            # scalar tensor
            Q_star_batch = Q_star.repeat(B)
        elif Q_star.ndim == 1:
            if Q_star.shape[0] != B:
                raise ValueError(f"Q_star has shape {Q_star.shape}, expected ({B},)")
            Q_star_batch = Q_star
        else:
            raise ValueError(f"Invalid Q_star ndim={Q_star.ndim}, expected scalar or (B,)")
    else:
        # numeric scalar
        Q_star_batch = torch.tensor([Q_star]*B, dtype=dQ.dtype, device=dQ.device)

    # --- Allocate output ---
    Q = torch.zeros_like(dQ)

    # --- Set reference point ---
    Q[:, idx] = Q_star_batch

    # --- Forward integrate (idx → right) ---
    for j in range(idx+1, J):
        dp = p[:, j] - p[:, j-1]
        Q[:, j] = Q[:, j-1] + 0.5 * (dQ[:, j] + dQ[:, j-1]) * dp

    # --- Backward integrate (idx → left) ---
    for j in range(idx-1, -1, -1):
        dp = p[:, j+1] - p[:, j]
        Q[:, j] = Q[:, j+1] - 0.5 * (dQ[:, j+1] + dQ[:, j]) * dp

    return Q.squeeze(0) if single else Q


def lqd(Q, p_grid, min_dQ=1e-6):
    dQ = differentiate(Q, p_grid)
    dQ = torch.clamp(dQ, min=min_dQ)
    return torch.log(dQ)


def inv_lqd(G, p_grid, p_star, Q_star):
    dQ = torch.exp(G)
    Q = integrate(dQ, p_grid, p_star, Q_star)
    return Q
