import torch


def project_trajectories(x_hat_norm, mean=None, std=None, mapping=None, dt=0.1, eps=1e-6):
    """
    Project generated trajectories onto a simple kinematic manifold:
    - Keep positions as produced (x,y)
    - Recompute speeds from finite differences of positions: speed_t = ||p_{t+1}-p_t|| / dt
    - Recompute accelerations from finite differences of speeds: a_t = (v_{t+1}-v_t) / dt

    This is conservative: it enforces kinematic consistency without altering
    the spatial path. It expects features to include position coords and
    speed/acc channels; indices are provided via `mapping`.

    Args:
        x_hat_norm: Tensor (N, K, L) normalized samples (float)
        mean: Tensor (K,) or None used for denormalization
        std: Tensor (K,) or None used for denormalization
        mapping: dict with keys 'pos_x', 'pos_y', 'vel', 'acc' mapping to int indices
        dt: float time step between timesteps (same units used in dataset)
        eps: small value to avoid division by zero

    Returns:
        Tensor (N, K, L) of corrected (normalized) samples
    """
    if not torch.is_tensor(x_hat_norm):
        return x_hat_norm

    device = x_hat_norm.device
    x = x_hat_norm.clone().to(dtype=torch.float32)

    # quick checks
    if mapping is None:
        return x

    pos_x_i = mapping.get("pos_x")
    pos_y_i = mapping.get("pos_y")
    vel_i = mapping.get("vel")
    acc_i = mapping.get("acc")

    if pos_x_i is None or pos_y_i is None or vel_i is None:
        # nothing to do
        return x

    # denormalize if mean/std provided
    if (mean is not None) and (std is not None):
        mean_t = mean.to(device=device).view(1, -1, 1)
        std_t = std.to(device=device).view(1, -1, 1)
        x_den = x * std_t + mean_t
    else:
        x_den = x

    # x_den: (N, K, L)
    N, K, L = x_den.shape

    # extract positions
    px = x_den[:, pos_x_i, :]  # (N, L)
    py = x_den[:, pos_y_i, :]

    # compute displacements between consecutive timesteps
    # forward difference for instantaneous speed at t -> use diff over dt
    # compute speed for t=0..L-2, pad last timestep with same speed
    dx = px[:, 1:] - px[:, :-1]
    dy = py[:, 1:] - py[:, :-1]
    dist = torch.sqrt(dx * dx + dy * dy + eps)
    v_from_pos = dist / float(dt)  # (N, L-1)

    # Make v_full shape (N, L): we set last timestep speed equal to last diff
    v_full = torch.zeros((N, L), device=device, dtype=x_den.dtype)
    if L >= 2:
        v_full[:, :-1] = v_from_pos
        v_full[:, -1] = v_from_pos[:, -1]

    # acceleration from speeds
    if acc_i is not None:
        dv = v_full[:, 1:] - v_full[:, :-1]
        a_from_v = dv / float(dt)
        a_full = torch.zeros((N, L), device=device, dtype=x_den.dtype)
        if L >= 2:
            a_full[:, :-1] = a_from_v
            a_full[:, -1] = a_from_v[:, -1]

    # write back computed quantities into denormalized tensor
    x_den_proj = x_den.clone()
    x_den_proj[:, vel_i, :] = v_full
    if acc_i is not None:
        x_den_proj[:, acc_i, :] = a_full

    # renormalize
    if (mean is not None) and (std is not None):
        x_proj = (x_den_proj - mean_t) / std_t
    else:
        x_proj = x_den_proj

    return x_proj
