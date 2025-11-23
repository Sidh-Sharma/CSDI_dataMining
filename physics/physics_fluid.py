import torch
import torch.nn.functional as F
import math


def compute_physics_loss(model, coords):
    """
    Compute physics loss (continuity and momentum residuals).

    Args:
        model: Trained model.
        coords: Coordinate tensor.

    Returns:
        Physics loss value.
    """
    coords.requires_grad_(True)
    pred = model(coords)
    u, v, p = pred[..., 0], pred[..., 1], pred[..., 2]

    # Continuity: du/dx + dv/dy = 0
    du_dx = torch.autograd.grad(u.sum(), coords, create_graph=True)[0][..., 0]
    dv_dy = torch.autograd.grad(v.sum(), coords, create_graph=True)[0][..., 1]
    continuity = (du_dx + dv_dy) ** 2

    # Momentum (simplified, steady-state, Re=1)
    du_dy = torch.autograd.grad(u.sum(), coords, create_graph=True)[0][..., 1]
    d2u_dy2 = torch.autograd.grad(du_dy.sum(), coords, create_graph=True)[0][..., 1]
    dp_dx = torch.autograd.grad(p.sum(), coords, create_graph=True)[0][..., 0]
    momentum_x = (d2u_dy2 + dp_dx) ** 2  # Simplified NS: nu*d2u/dy2 = dp/dx, nu=1

    physics_loss = torch.mean(continuity + momentum_x)
    return physics_loss


def compute_boundary_loss(model, coords):
    """
    Compute boundary loss for no-slip conditions.

    Args:
        model: Trained model.
        coords: Coordinate tensor.

    Returns:
        Boundary loss value.
    """
    # Boundaries at y = ±0.5
    mask_top = torch.abs(coords[..., 1] - 0.5) < 1e-3
    mask_bottom = torch.abs(coords[..., 1] + 0.5) < 1e-3
    mask = mask_top | mask_bottom

    pred = model(coords)
    u, v = pred[..., 0], pred[..., 1]
    boundary_u = torch.mean((u[mask]) ** 2) if mask.any() else torch.tensor(0.0, device=pred.device)
    boundary_v = torch.mean((v[mask]) ** 2) if mask.any() else torch.tensor(0.0, device=pred.device)

    return boundary_u + boundary_v


def physics_loss_fn(x_hat, mean=None, std=None, dt=0.1):
    """
    Dataset-facing wrapper for fluid Kaggle physics loss.

    At present this function is a safe placeholder: it returns zero so that
    enabling `use_physics` does not break training. The detailed physics
    functions `compute_physics_loss` and `compute_boundary_loss` are provided
    above and can be adapted to your coordinate parametrization and a true
    PDE-model that maps coordinates -> (u,v,p).

    Args:
        x_hat: tensor (B, K, L) representing reconstructed fields.
        mean, std: optional normalization tensors (unused currently).
        dt: time-step / physical parameter (unused currently).

    Returns:
        A torch scalar tensor (physics loss).
    """
    # TODO: implement mapping from x_hat -> continuous model(coords) and use
    # compute_physics_loss / compute_boundary_loss. For now, return zero so
    # training remains stable.
    # Expect x_hat shape: (B, K, L) where K equals 3 * (H*W)
    if not isinstance(x_hat, torch.Tensor):
        return torch.tensor(0.0, dtype=torch.float32)

    device = x_hat.device

    if x_hat.dim() != 3:
        return torch.tensor(0.0, dtype=torch.float32, device=device)

    B, K, L = x_hat.shape

    # Require at least 3 channels per spatial point (u,v,p)
    if K < 3:
        return torch.tensor(0.0, dtype=torch.float32, device=device)

    # Try to interpret K as 3 * (H*W)
    if K % 3 != 0:
        return torch.tensor(0.0, dtype=torch.float32, device=device)

    spatial_points = K // 3
    # try square grid first
    H = int(round(math.sqrt(spatial_points)))
    if H * H == spatial_points:
        W = H
    else:
        # fall back to single-row grid
        H = 1
        W = spatial_points

    try:
        # reshape to (B, 3, H, W, L) then permute to (B, L, 3, H, W)
        x = x_hat.view(B, 3, H, W, L).permute(0, 4, 1, 2, 3)  # (B, L, 3, H, W)
    except Exception:
        return torch.tensor(0.0, dtype=torch.float32, device=device)

    # work with floats and enable gradients through x
    x = x.to(dtype=torch.float32)

    # separate channels
    u = x[:, :, 0, :, :]  # (B, L, H, W)
    v = x[:, :, 1, :, :]
    p = x[:, :, 2, :, :]

    # central differences with replicate padding
    # du/dx (x along W axis)
    u_pad_x = F.pad(u, (1, 1, 0, 0), mode="replicate")
    du_dx = (u_pad_x[:, :, :, 2:] - u_pad_x[:, :, :, :-2]) / 2.0

    # dv/dy (y along H axis)
    v_pad_y = F.pad(v, (0, 0, 1, 1), mode="replicate")
    dv_dy = (v_pad_y[:, :, 2:, :] - v_pad_y[:, :, :-2, :]) / 2.0

    # continuity residual
    continuity = (du_dx + dv_dy) ** 2

    # momentum: d2u/dy2 + dp/dx
    # first derivative du/dy
    u_pad_y = F.pad(u, (0, 0, 1, 1), mode="replicate")
    du_dy = (u_pad_y[:, :, 2:, :] - u_pad_y[:, :, :-2, :]) / 2.0
    # second derivative d2u/dy2
    du_dy_pad = F.pad(du_dy, (0, 0, 1, 1), mode="replicate")
    d2u_dy2 = (du_dy_pad[:, :, 2:, :] - du_dy_pad[:, :, :-2, :]) / 2.0

    # dp/dx
    p_pad_x = F.pad(p, (1, 1, 0, 0), mode="replicate")
    dp_dx = (p_pad_x[:, :, :, 2:] - p_pad_x[:, :, :, :-2]) / 2.0

    momentum_x = (d2u_dy2 + dp_dx) ** 2

    # mean over spatial points and time and batch
    phys_residual = torch.mean(continuity + momentum_x)

    # boundary loss: enforce u,v ~ 0 at top/bottom rows
    if H >= 2:
        top = u[:, :, 0, :]
        bottom = u[:, :, -1, :]
        boundary_u = torch.mean(top ** 2) + torch.mean(bottom ** 2)

        top_v = v[:, :, 0, :]
        bottom_v = v[:, :, -1, :]
        boundary_v = torch.mean(top_v ** 2) + torch.mean(bottom_v ** 2)
    else:
        # no meaningful boundary in H==1 case
        boundary_u = torch.tensor(0.0, device=device)
        boundary_v = torch.tensor(0.0, device=device)

    boundary_loss = boundary_u + boundary_v

    total_loss = phys_residual + boundary_loss
    print(total_loss)
    return total_loss
