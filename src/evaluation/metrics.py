"""
Metrics list:
    RE  — Residual-divergence MSE (discrete divergence residual)
    WS  — Sliced Wasserstein Distance (via POT)
    JS  — Jensen-Shannon divergence (via scipy)
    NSR — Incompressible Navier-Stokes residual (vorticity form, §11)
"""

from typing import Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
#  1. Residual-divergence MSE  (RE)
# ---------------------------------------------------------------------------

def divergence_spectral(velocity: torch.Tensor, domain_size: float = 2.0 * np.pi) -> torch.Tensor:
    """Per-sample divergence field via spectral derivatives.

    Matches the convention of PeriodicStreamFunctionCodec
    (x-axis = dim 2/H, y-axis = dim 3/W):
        div̂(k) = j ω_x · û_x(k)  +  j ω_y · û_y(k)
    with ω_x along dim 2 and ω_y along dim 3.

    Args:
        velocity:   [B, 2, H, W]  float32
        domain_size: physical side length (default 2π).

    Returns:
        divergence: [B, 1, H, W]  real-space divergence.
    """
    B, C, H, W = velocity.shape
    assert C == 2, f"Expected 2-channel velocity, got {C}"

    vx_hat  = torch.fft.fft2(velocity[:, 0])
    vy_hat  = torch.fft.fft2(velocity[:, 1])

    # Angular spatial frequency (matching PeriodicStreamFunctionCodec)
    dx      = domain_size / W
    dy      = domain_size / H
    omega_x = 2.0 * np.pi * torch.fft.fftfreq(H, d=dy, device=velocity.device)  # dim 2
    omega_y = 2.0 * np.pi * torch.fft.fftfreq(W, d=dx, device=velocity.device)  # dim 3
    omega_x = omega_x.view(1, H, 1)
    omega_y = omega_y.view(1, 1, W)

    div_hat = 1j * omega_x * vx_hat + 1j * omega_y * vy_hat
    return torch.fft.ifft2(div_hat).real.unsqueeze(1)