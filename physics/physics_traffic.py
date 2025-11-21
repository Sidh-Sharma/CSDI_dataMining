import torch
import torch.nn.functional as F

def physics_loss_soft(
    x_hat,
    mean,
    std,
    target_mask=None,
    pos_idx=0,
    vel_idx=2,
    acc_idx=3,
    dt=0.1,
    eps=1e-9,
):
    """
    Soft physics loss for traffic, normalized-space formulation.

    Inputs:
      x_hat:      Tensor (B, K, L) - model's reconstructed state in NORMALIZED units
      mean:       1D Tensor or array (K,) - training means (raw units)
      std:        1D Tensor or array (K,)  - training stds (raw units)
      target_mask: optional Tensor (B, K, L) of {0,1} indicating positions to evaluate.
                   If None, all positions are used.
      pos_idx, vel_idx, acc_idx: integer feature indices in K for position, velocity, acceleration
      dt:         timestep in seconds (raw units)
      eps:        small number for numerical stability

    Returns:
      scalar tensor (differentiable) = normalized soft physics loss (mean over valid terms)
    -----------------------------------------------------------------------------
    Key idea:
      - Work in normalized coordinates (x_hat). Finite differences in raw units are:
          Δpos_raw / dt = implied velocity (raw)
        but Δpos_raw = Δpos_norm * std_pos, since pos_raw = pos_norm * std_pos + mean_pos.
      - Construct target for normalized velocity:
          vel_norm_target = (Δpos_norm * std_pos / dt - mean_vel) / std_vel
        and compare to predicted vel_norm.
      - Similarly for acceleration:
          acc_norm_target = (Δvel_norm * std_vel / dt - mean_acc) / std_acc
      - This yields dimensionless residuals of order ~1 and compatible with diffusion loss.
    -----------------------------------------------------------------------------
    """
    device = x_hat.device

    # ensure mean/std are tensors on correct device
    if not torch.is_tensor(mean):
        mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    else:
        mean_t = mean.to(device).float()
    if not torch.is_tensor(std):
        std_t = torch.tensor(std, dtype=torch.float32, device=device)
    else:
        std_t = std.to(device).float()

    B, K, L = x_hat.shape
    assert 0 <= pos_idx < K and 0 <= vel_idx < K and 0 <= acc_idx < K

    # optional mask: use all ones if not provided
    if target_mask is None:
        mask = torch.ones((B, K, L), dtype=x_hat.dtype, device=device)
    else:
        mask = target_mask.to(device).float()

    # Extract normalized channels (B, L)
    pos_norm = x_hat[:, pos_idx, :]    # (B, L)
    vel_norm = x_hat[:, vel_idx, :]    # (B, L)
    acc_norm = x_hat[:, acc_idx, :]    # (B, L)

    # per-feature scalars
    std_pos = std_t[pos_idx].clamp(min=eps)
    std_vel = std_t[vel_idx].clamp(min=eps)
    std_acc = std_t[acc_idx].clamp(min=eps)

    mean_vel = mean_t[vel_idx]
    mean_acc = mean_t[acc_idx]

    # compute finite differences in normalized space
    # Δpos_norm = pos_norm[:, 1:] - pos_norm[:, :-1]  -> corresponds to Δpos_raw/std_pos
    if L < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    delta_pos_norm = pos_norm[:, 1:] - pos_norm[:, :-1]    # (B, L-1)
    delta_vel_norm = vel_norm[:, 1:] - vel_norm[:, :-1]    # (B, L-1)

    # Construct normalized targets for velocity and acceleration:
    # vel_norm_target = (Δpos_raw / dt - mean_vel) / std_vel
    # Δpos_raw = Δpos_norm * std_pos
    vel_norm_target = (delta_pos_norm * std_pos / (dt + eps) - mean_vel) / (std_vel + eps)   # (B, L-1)

    # acc_norm_target = (Δvel_raw / dt - mean_acc) / std_acc
    # Δvel_raw = Δvel_norm * std_vel
    acc_norm_target = (delta_vel_norm * std_vel / (dt + eps) - mean_acc) / (std_acc + eps)   # (B, L-1)

    # Predictions aligned: we compare vel_norm[:, :-1] to vel_norm_target (both length L-1)
    vel_pred_aligned = vel_norm[:, :-1]   # (B, L-1)
    acc_pred_aligned = acc_norm[:, :-1]   # (B, L-1)

    # create evaluation masks aligned to same timesteps (use logical AND to ensure data present)
    mask_pos = mask[:, pos_idx, :-1]      # (B, L-1) - requires pos at t and t+1
    mask_vel = mask[:, vel_idx, :-1]      # (B, L-1) - requires vel at t
    mask_acc = mask[:, acc_idx, :-1]      # (B, L-1) - requires acc at t

    # combine masks: evaluate vel consistency only where pos and vel are observed
    mask_vel_cons = (mask_pos * mask_vel).float()   # (B, L-1)
    # evaluate acc consistency only where vel and acc are observed
    mask_acc_cons = ((mask[:, vel_idx, :-1] * mask_acc)).float()   # (B, L-1)

    # Residuals
    resid_vel = vel_pred_aligned - vel_norm_target   # (B, L-1)
    resid_acc = acc_pred_aligned - acc_norm_target   # (B, L-1)

    # Apply masks and compute mean squared residual over valid entries
    valid_vel = mask_vel_cons.sum()
    valid_acc = mask_acc_cons.sum()

    if valid_vel.item() > 0:
        # squared residual weighted by mask
        sq_vel = (resid_vel ** 2) * mask_vel_cons
        L_vel = sq_vel.sum() / (valid_vel + eps)
    else:
        L_vel = torch.tensor(0.0, device=device, requires_grad=True)

    if valid_acc.item() > 0:
        sq_acc = (resid_acc ** 2) * mask_acc_cons
        L_acc = sq_acc.sum() / (valid_acc + eps)
    else:
        L_acc = torch.tensor(0.0, device=device, requires_grad=True)

    # return sum (you can scale externally by lambda_phys)
    return L_vel + L_acc
