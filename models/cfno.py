"""
Why FNO for our task:
    - Kolmogorov flow lives on a 2*pi periodic torus; 
      FNO's spectral convolution is exactly periodic by construction (no patch boundary artefacts that hurt the divergence in DiT).
    - Energy spectrum of forced 2D turbulence concentrates at low/medium wavenumbers, which matches FNO's truncated-mode design.
    - Spectral mixing is global with O(N log N) cost; gives long-range receptive field without depth.
    - Truncated modes act as a low-pass filter -> output is smooth, 
      derivatives (used in physics_layer for hard constraint) are numerically stable.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def _sinusoidal_time_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
    half    = dim // 2
    freqs   = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args    = t.float()[:, None] * freqs[None]
    emb     = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

class SpectralConv2d(nn.Module):
    """Truncated 2D spectral convolution.

    Multiplies the lowest (modes1, modes2) Fourier coefficients with a learnable complex weight.
    """
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.modes1         = modes1
        self.modes2         = modes2

        scale               = 1.0 / (in_channels * out_channels)
        # We split complex weights into real and imaginary parts to support DDP/NCCL
        self.weights1_real  = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))

        self.weights1_imag  = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))

        self.weights2_real  = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))

        self.weights2_imag  = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))

    @property
    def weights1(self):
        return torch.complex(self.weights1_real, self.weights1_imag)

    @property
    def weights2(self):
        return torch.complex(self.weights2_real, self.weights2_imag)

    @staticmethod
    def _compl_mul2d(x, w):
        """
        x: (B, in_c, M1, M2), w: (in_c, out_c, M1, M2) -> (B, out_c, M1, M2)
        """
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def apply_ft(self, x_ft, m1=None, m2=None):
        """Truncated spectral multiplication

        Only the two blocks of retained modes are filled (the upper block with kx>=0 and the lower block with kx<0 in the rfft layout), 
        and all other frequency points are set to zero.
        
        It does not include the FFT, so it can be reused by downstream modules that need intermediate spectral results (e.g., Mode-wise FiLM).
        
        x_ft [B, in, H, W//2+1] -> [B, out, H, W//2+1].
        """
        B, _, H, W2             = x_ft.shape
        m1                      = min(self.modes1, H // 2, m1 if m1 is not None else self.modes1)
        m2                      = min(self.modes2, W2,     m2 if m2 is not None else self.modes2)
        out_ft                  = torch.zeros(B, self.out_channels, H, W2, dtype=torch.cfloat, device=x_ft.device)

        out_ft[:, :, :m1, :m2]  = self._compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])

        out_ft[:, :, -m1:, :m2] = self._compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])

        return out_ft

    def forward(self, x):
        B, _, H, W  = x.shape
        out_ft      = self.apply_ft(torch.fft.rfft2(x, norm="ortho"))
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOBlock(nn.Module):
    """
    Standard FNO block: spectralconv2d + pointwise(1x1 conv) + GELU, 
    optionally with FiLM-style modulation by a timestep embedding and/or a physical-parameter embedding.
    """
    def __init__(self, channels: int, modes1: int, modes2: int, time_dim: int = 0,
                 param_dim: int = 0):
        super().__init__()
        self.spectral       = SpectralConv2d(channels, channels, modes1, modes2)
        self.pointwise      = nn.Conv2d(channels, channels, kernel_size=1)
        self.act            = nn.GELU()

        self.time_dim       = time_dim
        self.param_dim      = param_dim

        # t_emb and p_emb both have time_dim channels after their MLPs.
        self.cond_dim       = time_dim + (time_dim if param_dim > 0 else 0)
        if self.cond_dim > 0:
            self.cond_proj  = nn.Linear(self.cond_dim, 2 * channels)
            nn.init.zeros_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x, t_emb=None, p_emb=None):
        """
        x_  -> .
        """
        h                       = self.spectral(x) + self.pointwise(x)
        # Film module
        if self.cond_dim > 0:
            cond_list           = [e for e in (t_emb, p_emb) if e is not None]
            if cond_list:
                cond            = torch.cat(cond_list, dim=-1)
                scale, shift    = self.cond_proj(cond).chunk(2, dim=-1)
                scale           = scale.unsqueeze(-1).unsqueeze(-1)
                shift           = shift.unsqueeze(-1).unsqueeze(-1)
                h               = h * (1 + scale) + shift
        return self.act(h)


# =============================================================================
# Mode-wise FiLM
# =============================================================================

class ModeWiseFilm2d(nn.Module):
    """Mode-wise FiLM: Condition-dependent mode-wise diagonal spectral modulation.

        $\hat{y}_o(k) = (1 + \gamma_o(k;c)) ⊙ \hat{z}_o(k) [+ \beta_o(k;c)]$

        Both $\gamma$ and $\beta$ depend on the condition c (e.g., timestep, physical parameters), but $\beta$ is not enabled:
        - $\gamma$ is real-valued and acts on the two kx half-blocks stored by rfft2 (the upper block with $kx≥0$ and the lower block with $kx<0$, where row u of the lower block is paired with row M1-u of the upper block); the output after irfft2 is always real-valued;
        - It is not a strictly "physically symmetric" operator: the gain is defined on the stored half-spectrum and does not guarantee $\gamma(-k)=\gamma(k)$ (i.e., not radial/even-symmetric), so it is not a strictly k-even-symmetric physical operator and does not fully guarantee Hermitian consistency; hence it is a "restricted mode gain/basis operator".
        - It is generated by a low-rank factorization: $\gamma_o(k;c) = \sum_r a_{o,r}(c) \psi_r(k)$, which keeps the parameter count small, controllable, and low-rank.
    """
    def __init__(self, channels: int, modes1: int, modes2: int,
                 cond_dim: int, rank: int = 8, use_beta: bool = False,
                 per_slot: int = 1, gamma_max: float | None = None):
        super().__init__()
        self.channels   = channels
        self.modes1     = modes1
        self.modes2     = modes2
        self.cond_dim   = cond_dim
        self.rank       = rank
        self.use_beta   = use_beta
        self.per_slot   = per_slot
        assert gamma_max is None or gamma_max > 0, f"gamma_max 必须 > 0(6.txt §三.3),got {gamma_max}"
        self.gamma_max  = gamma_max

        # Learnable mode basis $\phi_r(k)$: $[R,\ M_1+1,\ M_2]$, row index = $kx \in [0,\ M_1]$, 
        # including the $kx=M_1$ row paired with the lower half-block; 
        # when rank=0, the modulation is disabled ($\gamma_o(k;c)=1$, i.e., identity).
        self.phi        = (nn.Parameter(0.1 * torch.randn(rank, modes1 + 1, modes2)) if rank > 0 else None)

        # Zero-initialize the a / β condition projections -> identity modulation at initialization,
        # degenerates to the unconditional FNO, stabilizing training,
        # all conditional paths are zero-initialized -> initially equivalent to an unconditional FNO block.
        self.a_proj     = (nn.Linear(cond_dim, per_slot * channels * rank) if rank > 0 else None)
        if self.a_proj is not None:
            nn.init.zeros_(self.a_proj.weight)
            nn.init.zeros_(self.a_proj.bias)
        self.b_proj     = None # β is disabled by default
        if use_beta and rank > 0:
            self.b_proj = nn.Linear(cond_dim, per_slot * channels * rank)
            nn.init.zeros_(self.b_proj.weight)
            nn.init.zeros_(self.b_proj.bias)

    @staticmethod
    def _gain_halves(grid):
        """Build the upper/lower half-block gains from the gain grid.
            grid: [B, G, C, M1+1, M2], row index = kx.
            Upper half: kx >= 0, use grid as is.
            Lower half: kx < 0, don't generate separately — reuse the upper half with a kx flip:
            - lower row u (kx = u-M1 < 0) pairs with upper row M1-u (flipped);
            - i.e. lower gains = first M1 rows of the row-flipped grid.
            - Note: u=0 with M1=H/2 is the Nyquist row, handle with care.
            Note: flip kx only; ky keeps the same index as the stored half-spectrum a "restricted mode gain".
        """
        M1 = grid.shape[-2] - 1
        return grid[:, :, :, :M1, :], torch.flip(grid, dims=[-2])[:, :, :, :M1, :]

    def forward(self, z_top, z_bot, cond):
        """z_top / z_bot: [B, C, M1', M2'] or [B, G, C, M1', M2'], 
        truncated spectral blocks, optionally with a G dimension (group/slot dimension), compatible with slots.
        """
        if self.rank == 0:
            return z_top, z_bot
        is4d = z_top.ndim == 4
        if is4d:
            z_top, z_bot = z_top.unsqueeze(1), z_bot.unsqueeze(1)
        B, G, C, M1, M2 = z_top.shape

        assert G == self.per_slot, f"Input G={G} is inconsistent with the constructed per_slot={self.per_slot}"

        M1 = min(self.modes1, M1)
        M2 = min(self.modes2, M2)
        phi = self.phi[:, :M1 + 1, :M2]

        a                   = self.a_proj(cond).reshape(B, G, C, self.rank)
        grid                = torch.einsum("bgcr, rxy -> bgcxy", a, phi)  # [B, G, C, M1+1, M2]
        if self.gamma_max is not None:                              # when gamma_max is not None
            grid            = self.gamma_max * torch.tanh(grid)     # Bounded gain: $γ = γ_{max}·tanh(γ̃)$
        g_top, g_bot        = self._gain_halves(grid)
        z_top               = z_top * (1 + g_top)
        z_bot               = z_bot * (1 + g_bot)
        if self.use_beta:
            b               = self.b_proj(cond).reshape(B, G, C, self.rank)
            bg              = torch.einsum("bgcr, rxy -> bgcxy", b, phi)
            bt, bb          = self._gain_halves(bg)
            z_top           = z_top + bt
            z_bot           = z_bot + bb
        if is4d:
            z_top, z_bot    = z_top.squeeze(1), z_bot.squeeze(1)
        return z_top, z_bot


class FNOBlockModeFilm(nn.Module):
    """FNO block + Mode-wise FiLM

        h = GELU( irfft2( module( W(k) ⊛ x̂(k) ) ) + W_1(x) )
    """

    def __init__(self, channels: int, modes1: int, modes2: int,
                 time_dim: int = 0, param_dim: int = 0,
                 film_rank: int = 8, use_beta: bool = False,
                 gamma_max: float | None = None):
        super().__init__()
        self.channels   = channels
        self.pointwise  = nn.Conv2d(channels, channels, kernel_size=1)
        self.act        = nn.GELU()
        self.time_dim   = time_dim
        self.param_dim  = param_dim
        # Both t_emb and p_emb, its dim is all the time_dim, so concat is 2xtime_dim
        self.cond_dim   = time_dim + (time_dim if param_dim > 0 else 0)
        self.spectral   = SpectralConv2d(channels, channels, modes1, modes2)
        
        # mode-wise FiLM(when film_rank=0 is equivalent)
        self.mode_film  = ModeWiseFilm2d(channels, modes1, modes2, self.cond_dim, rank=film_rank, use_beta=use_beta, gamma_max=gamma_max)

    def forward(self, x, t_emb=None, p_emb=None):
        
        B, _, H, W  = x.shape
        cond_list   = [e for e in (t_emb, p_emb) if e is not None]
        if cond_list:
            cond    = torch.cat(cond_list, dim=-1)
        else:
            cond    = torch.zeros(B, self.cond_dim, device=x.device, dtype=x.dtype)

        # FFT
        x_ft        = torch.fft.rfft2(x, norm="ortho")          # [B, C, H, W//2+1]
        
        # Truncated Spectral Convolution
        out_ft      = self.spectral.apply_ft(x_ft)              
        
        # ModeWiseFilm2d
        m1          = min(self.spectral.modes1, H // 2)
        m2          = min(self.spectral.modes2, W // 2 + 1)
        top, bot    = self.mode_film(out_ft[:, :, :m1, :m2], out_ft[:, :, -m1:, :m2], cond)

        # Assemble with a new tensor
        # do not write back to out_ft: top/bot are views of it, and in-place writes will break the backward graph
        q_ft        = torch.zeros_like(out_ft)
        q_ft[:, :, :m1, :m2]    = top
        q_ft[:, :, -m1:, :m2]   = bot
        q           = torch.fft.irfft2(q_ft, s=(H, W), norm="ortho")

        # act(fconv + pwconv)
        return self.act(q + self.pointwise(x))


class FNO2d(nn.Module):
    """    
    Backbone:
        Lift (1x1 conv): in_channels   -> width
        N x FNOBlock   : width         -> width
        Project        : width         -> 128 -> out_channels (two 1x1 convs + GELU)

    Conditioning on low-res input is done by bilinear-upsampling y to the target
    resolution and concatenating along the channel dimension before the lift.
    """
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        width: int = 64,
        depth: int = 4,
        modes: int = 32,
        time_dim: int = 128,
        use_lr_shortcut: bool = True,
        cond_in_channels: int = 2,
    ):
        super().__init__()
        self.in_channels        = in_channels
        self.cond_in_channels   = cond_in_channels
        self.out_channels       = out_channels
        self.use_lr_shortcut    = use_lr_shortcut
        self.time_dim           = time_dim

        # Time/timestep embedding (used only when timesteps is not None).
        self.t_mlp              = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Lift: x and upsampled y are concatenated -> 2 * cond_in_channels channels (or in+cond).
        lift_in                 = in_channels + cond_in_channels + 2  # +2 for grid (x, y) coords
        self.lift               = nn.Conv2d(lift_in, width, kernel_size=1)

        self.blocks             = nn.ModuleList([FNOBlock(width, modes, modes, time_dim=time_dim) for _ in range(depth)])

        self.proj               = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=1),
        )

        # Initialize the final projection close to zero so the residual shortcut
        # dominates at the start of training (huge inductive bias for SR).
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    @staticmethod
    def _grid(B: int, H: int, W: int, device, dtype):
        gx      = torch.arange(H, device=device, dtype=dtype) / H
        gy      = torch.arange(W, device=device, dtype=dtype) / W
        gx, gy  = torch.meshgrid(gx, gy, indexing='ij')
        grid    = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return grid

    def forward(self, x, timesteps=None, y=None, **kwargs):
        if y is None:
            raise ValueError("FNO2d requires low-res conditioning `y`.")

        # Determine target resolution.
        if x is not None:
            target_h, target_w  = x.shape[2], x.shape[3]
        else:
            # If no x is given (pure regression), infer from y.
            scale               = kwargs.get("scale", 4)
            target_h, target_w  = y.shape[2] * scale, y.shape[3] * scale
            x                   = torch.zeros(y.shape[0], self.in_channels, target_h, target_w, device=y.device, dtype=y.dtype)

        B           = x.shape[0]
        y_up        = F.interpolate(y, size=(target_h, target_w), mode='bilinear', align_corners=False)
        grid        = self._grid(B, target_h, target_w, x.device, x.dtype)

        h           = torch.cat([x, y_up, grid], dim=1)
        h           = self.lift(h)

        # Timestep embedding
        if timesteps is None:
            t_emb   = None
        else:
            t_emb   = _sinusoidal_time_embedding(timesteps, self.time_dim)
            t_emb   = self.t_mlp(t_emb)

        for blk in self.blocks:
            h       = blk(h, t_emb)

        out         = self.proj(h)

        # Residual SR shortcut: only valid when out_channels matches in_channels.
        if self.use_lr_shortcut and self.out_channels == self.cond_in_channels:
            out     = out + y_up
        return out


# =============================================================================
# Parametric-conditioned FNO vector field (Task A: feasible prior Flow Matching)
# =============================================================================

class FNO2dParam(nn.Module):
    """
    Conditioned FNO vector field for Flow Matching on divergence-free stream-function (psi) states.

    Difference from FNO2d (super-resolution):
      - No low-res conditioning `y`; conditioning is a low-dimensional physical-parameter vector `parameters` (e.g. Re), 
        embedded by an MLP and injected into every FNO block via FiLM.
      - Output is the FM vector field in psi-space, forced to zero-mean so the ODE trajectory stays in the feasible (gauge-fixed) tangent space.
      - SpectralConv2d's truncated modes act as an intrinsic low-pass filter,
        which suppresses the high-frequency noise amplification that a plain CNN suffers from when decoding psi -> velocity.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 64,
        depth: int = 4,
        modes: int = 16,
        time_dim: int = 128,
        param_dim: int = 1,
        zero_mean_output: bool = True,
        film_kind: str = "global",
        film_rank: int = 8,
        use_beta: bool = False,
        gamma_max: float | None = None,
        kernel_kind: str = "none",
        kernel_rank: int = 4):
        super().__init__()

        assert film_kind in ("none", "global", "mode"), \
            f"未知 film_kind: {film_kind}"
        assert kernel_kind in ("none", "cond_kernel"), \
            f"未知 kernel_kind: {kernel_kind}"
        self.in_channels        = in_channels
        self.out_channels       = out_channels
        self.width              = width
        self.depth              = depth
        self.modes              = modes
        self.time_dim           = time_dim
        self.param_dim          = param_dim
        self.zero_mean_output   = zero_mean_output
        self.film_kind          = film_kind
        self.film_rank          = film_rank
        self.use_beta           = use_beta
        self.gamma_max          = gamma_max
        self.kernel_kind        = kernel_kind
        self.kernel_rank        = kernel_rank

        # condition encoder branch, when film_kind="none" and kernel_kind="none" it degrads FNO-InputConcat baseline.
        needs_cond      = film_kind != "none" or kernel_kind == "cond_kernel"
        self.needs_cond = needs_cond
        self.t_mlp      = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        ) if needs_cond else None
        # Physical-parameter embedding (e.g. normalized Reynolds number).
        if param_dim > 0 and needs_cond:
            self.p_mlp  = nn.Sequential(nn.Linear(param_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim),)
        else:
            self.p_mlp  = None

        # Lift: x + grid coordinates (no low-res conditioning).
        lift_in         = in_channels + 2
        self.lift       = nn.Conv2d(lift_in, width, kernel_size=1)

        # blocks
        if kernel_kind == "mode":
            # mode-wise FiLM(Frequency-domain)
            block       = lambda: FNOBlockModeFilm(width, modes, modes, time_dim=time_dim, param_dim=param_dim, 
                                                   film_rank=film_rank, use_beta=use_beta, gamma_max=gamma_max)
        elif film_kind == "global":
            # channel-wise FiLM(time/spatial-domain)
            block       = lambda: FNOBlock(width, modes, modes, time_dim=time_dim, param_dim=param_dim)
        else:
            # No modulation (conditioning via input concatenation)
            block       = lambda: FNOBlock(width, modes, modes, time_dim=0, param_dim=0)
        self.blocks     = nn.ModuleList([block() for _ in range(depth)])

        # projection
        self.proj       = nn.Sequential(nn.Conv2d(width, 128, kernel_size=1), nn.GELU(), nn.Conv2d(128, out_channels, kernel_size=1))

        # Zero-init the final projection: the flow starts from the zero field.
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    @staticmethod
    def _grid(B: int, H: int, W: int, device, dtype):
        """Generate a normalized 2D coordinate grid of shape `(B, 2, H, W)`.
        """
        gx      = torch.arange(H, device=device, dtype=dtype) / H
        gy      = torch.arange(W, device=device, dtype=dtype) / W
        gx, gy  = torch.meshgrid(gx, gy, indexing='ij')
        grid    = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return grid

    def forward(self, x, timesteps=None, parameters=None):
        """Forward: (x, timesteps=None, parameters=None) -> dx  [B, out_channels, H, W]
        """
        B, _, H, W      = x.shape
        grid            = self._grid(B, H, W, x.device, x.dtype)
        h               = self.lift(torch.cat([x, grid], dim=1))

        if not self.needs_cond:
            # no modulation,the conditioning is injected via input channel concatenation (var FNO-InputConcat baseline).
            for blk in self.blocks:
                h       = blk(h)
        else:
            # FM pseudo-time embedding (accept both [B] and [B, 1]).
            if timesteps is None:
                t_emb   = torch.zeros(B, self.time_dim, device=x.device, dtype=x.dtype)
            else:
                if timesteps.ndim == 2:
                    timesteps = timesteps.squeeze(-1)
                t_emb   = self.t_mlp(_sinusoidal_time_embedding(timesteps, self.time_dim))

            # Physical-parameter embedding.
            if self.p_mlp is None:
                p_emb   = None
            elif parameters is None:
                p_emb   = torch.zeros(B, self.time_dim, device=x.device, dtype=x.dtype)
            else:
                p_emb   = self.p_mlp(parameters)

            # block forward
            for blk in self.blocks:
                h       = blk(h, t_emb=t_emb, p_emb=p_emb)

        # projection
        out             = self.proj(h)

        # Keep the FM vector field in the zero-mean psi tangent space,
        # only required for the psi stream-function space; disable this option for images/generic states.)
        if self.zero_mean_output:
            out         = out - out.mean(dim=(-2, -1), keepdim=True)
        return out


# =============================================================================
# Factory: regression and hard-constraint variants
# =============================================================================

# -----------------------------------------------------------------------------
# Parametric (Task A) variants
# -----------------------------------------------------------------------------

FNO_PARAM_CONFIGS = {
    "fno_s_param": dict(
        in_channels=1, out_channels=1,
        width=32, depth=4, modes=12, time_dim=128, param_dim=1,
    ),
    "fno_b_param": dict(
        in_channels=1, out_channels=1,
        width=64, depth=4, modes=16, time_dim=128, param_dim=1,
    ),
    "fno_l_param": dict(
        in_channels=1, out_channels=1,
        width=96, depth=6, modes=24, time_dim=128, param_dim=1,
    ),
}

def FNO_S_PARAM(**kwargs):
    cfg = dict(FNO_PARAM_CONFIGS["fno_s_param"]); cfg.update(kwargs)
    return FNO2dParam(**cfg)

def FNO_B_PARAM(**kwargs):
    cfg = dict(FNO_PARAM_CONFIGS["fno_b_param"]); cfg.update(kwargs)
    return FNO2dParam(**cfg)

def FNO_L_PARAM(**kwargs):
    cfg = dict(FNO_PARAM_CONFIGS["fno_l_param"]); cfg.update(kwargs)
    return FNO2dParam(**cfg)


# =============================================================================
# ConditionalFNO2D：条件向量场统一接口(theory/README.md §7.1 / §8)
# =============================================================================

class ConditionalFNO2D(nn.Module):
    """2D FNO 条件向量场 v_θ(x_s, s | c_t)，与 ConditionalUNet2D 接口完全一致。

    输入：
        x_s      [B, C, H, W]             FM 中间状态
        flow_time [B]                     FM 时间 s
        context  [B, K, C, H, W]          历史帧条件
    历史帧按 §8.1 合并为 K*C 通道后与 x_s 拼接(C + K*C + 可选 forcing/geometry)；
    输出与 x_s 完全同 shape(核心接口约束)。

    条件注入方式由 cond_mode 选择(13.txt §十三.A / 14.txt §四 的模型族):
      - "global_film"(M1, 默认): 标量条件 (s, dt, params) 经 sinusoidal
        embedding + channel-wise FiLM 注入每个 FNO block(s 作为 timesteps，
        dt 与 params 拼接后作为 parameters);
      - "mode_film"(M2, 阶段 1): 同上条件编码,但调制发生在频域、逐 mode:
        ŷ_o(k) = (1 + γ_o(k;c)) ⊙ (W_0(k) ⊛ x̂(k))_o [+ β_o(k;c)]，
        γ/β 由低秩 mode 基 φ_r(k) 生成,零初始化(初始退化为无条件 FNO);
      - "cond_kernel"(M3, 阶段 2): 直接条件化谱核本身:
        W(k;c) = W_0(k) + Σ_{r=1}^{R} a_r(c) W_r(k),a 零初始化 ->
        初始严格退化为固定核 FNO;核条件系数由 (s, dt, params) 生成
        (与 M1/M2 相同的条件编码);
      - "slot"(阶段 3 系列,帧通道化以外的主模型): 保留显式 slot 轴
        [B, T, C, H, W],只在 H, W 上做 2D FFT;底层为 FNO2dSlot,
        由 use_cond_kernel / use_mode_film / slot_rank 等 flag 选择
        M4(slot + 条件化谱核)/ M5(+ mode-wise FiLM)/ G 等变体;
      - "input_concat"(M0): 标量条件 (s, dt, params) 广播到空间网格后
        直接与输入拼接: u'(x) = [x_s, u_{t-K+1:t}(x), s, dt, params](x)，
        FNO block 不携带任何调制分支(底层 net 以 film_kind="none" 构建)。

    Args:
        state_channels: 状态通道数 C
        context_len:    历史帧数 K
        param_dim:      物理参数维度 P(global/mode_film/cond_kernel 时 dt 自动并入，p_mlp 输入为 P+1)
        extra_channels: 额外条件通道数(forcing + geometry_mask，§8.1 可选)
        width / modes / depth: FNO 配置(README §8.2 推荐 width=64, modes=20, depth=4)
        zero_mean_output: 强制输出零空间均值(psi 流函数空间专用；图像状态默认 False)
        block_len:      联合生成块长度 L(默认 1：单帧状态，等价原行为)。
                        状态 = 整块未来帧 [B, C*L, H, W]，输入通道 = C*(L+K)，
                        输出通道 = C*L。
        cond_mode:      "global_film"(M1)| "mode_film"(M2)|
                        "cond_kernel"(M3)| "slot"(阶段 3 系列)|
                        "input_concat"(M0)
        film_rank:      mode-wise FiLM 的 mode 基数量 R(mode_film / slot 模式)
        use_beta:       是否启用频域加性偏置 β(默认关闭,13.txt §7.2)
        kernel_rank:    条件化谱核基数量 R(cond_kernel / slot 模式);
        kernel_form:    "basis"(逐基核加权,默认)| "channel_low_rank"(U V^H)
        channel_low_rank: channel_low_rank 形式的秩 q(默认 in_channels//4)
        use_cond_kernel / use_mode_film / slot_rank / mode_basis /
        block_triangular / t0_init: 阶段 3 slot 模式 flag(见 FNO2dSlot)
    """

    def __init__(self, state_channels: int = 1, context_len: int = 4,
                 param_dim: int = 1, image_size: int = 64,
                 width: int = 64, modes: int = 20, depth: int = 4,
                 time_dim: int = 128, extra_channels: int = 0,
                 zero_mean_output: bool = False, block_len: int = 1,
                 cond_mode: str = "global_film",
                 film_rank: int = 8, use_beta: bool = False,
                 gamma_max: float | None = None,
                 kernel_rank: int = 4, kernel_form: str = "basis",
                 channel_low_rank: int | None = None,
                 kernel_coeff_max: float | None = None,
                 use_cond_kernel: bool = False, use_mode_film: bool = False,
                 slot_rank: int = 4, mode_basis: bool = True,
                 block_triangular: bool = True, t0_init: str = "identity"):
        super().__init__()
        assert cond_mode in ("input_concat", "global_film", "mode_film",
                             "cond_kernel", "mode_film_kernel",
                             "slot"), f"未知 cond_mode: {cond_mode}"
        self.state_channels = state_channels
        self.context_len = context_len
        self.block_len = block_len
        self.param_dim = param_dim
        self.extra_channels = extra_channels
        self.cond_mode = cond_mode
        self.film_rank = film_rank
        self.use_beta = use_beta
        self.gamma_max = gamma_max
        self.kernel_rank = kernel_rank
        self.kernel_form = kernel_form
        self.channel_low_rank = channel_low_rank
        self.kernel_coeff_max = kernel_coeff_max
        if cond_mode == "slot":
            # 阶段 3 系列: 显式 slot 轴主模型(M4/M5/G/显式 slot 基线)
            self.net = FNO2dSlot(
                state_channels=state_channels, context_len=context_len,
                block_len=block_len, param_dim=param_dim,
                image_size=image_size, width=width, modes=modes, depth=depth,
                time_dim=time_dim, extra_channels=extra_channels,
                zero_mean_output=zero_mean_output,
                use_cond_kernel=use_cond_kernel, kernel_rank=kernel_rank,
                kernel_form=kernel_form, channel_low_rank=channel_low_rank,
                use_mode_film=use_mode_film, film_rank=film_rank,
                use_beta=use_beta, gamma_max=gamma_max, slot_rank=slot_rank,
                mode_basis=mode_basis,
                block_triangular=block_triangular, t0_init=t0_init,
                kernel_coeff_max=kernel_coeff_max)
            return
        in_channels = state_channels * (block_len + context_len) + extra_channels
        if cond_mode == "input_concat":
            # M0: 广播标量条件 s / dt / params 各占 1 + 1 + P 个输入通道
            in_channels += param_dim + 2
            self.net = FNO2dParam(
                in_channels=in_channels,
                out_channels=state_channels * block_len,
                width=width,
                depth=depth,
                modes=modes,
                time_dim=time_dim,
                param_dim=0,             # 条件不进调制分支,全部走输入拼接
                film_kind="none",
                zero_mean_output=zero_mean_output,
            )
        elif cond_mode == "mode_film":
            # M2: 频域 mode-wise FiLM(阶段 1)
            self.net = FNO2dParam(
                in_channels=in_channels,
                out_channels=state_channels * block_len,
                width=width,
                depth=depth,
                modes=modes,
                time_dim=time_dim,
                param_dim=param_dim + 1,     # [dt, params] 拼接后作为参数条件
                film_kind="mode",
                film_rank=film_rank,
                use_beta=use_beta,
                gamma_max=gamma_max,
                kernel_coeff_max=kernel_coeff_max,
                zero_mean_output=zero_mean_output,
            )
        elif cond_mode == "cond_kernel":
            # M3: 条件化谱核(阶段 2)
            self.net = FNO2dParam(
                in_channels=in_channels,
                out_channels=state_channels * block_len,
                width=width,
                depth=depth,
                modes=modes,
                time_dim=time_dim,
                param_dim=param_dim + 1,     # [dt, params] 拼接后作为参数条件
                film_kind="none",
                kernel_kind="cond_kernel",
                kernel_rank=kernel_rank,
                kernel_form=kernel_form,
                channel_low_rank=channel_low_rank,
                kernel_coeff_max=kernel_coeff_max,
                gamma_max=gamma_max,
                zero_mean_output=zero_mean_output,
            )
        elif cond_mode == "mode_film_kernel":
            # 消融 D 有界版(18.txt 实验 1): ModeFiLM-Bnd + 条件化谱核
            # (film_kind="mode" + kernel_kind="cond_kernel",FNO2dParam
            #  已支持该组合,cfno.py 消融 D 注释)
            self.net = FNO2dParam(
                in_channels=in_channels,
                out_channels=state_channels * block_len,
                width=width,
                depth=depth,
                modes=modes,
                time_dim=time_dim,
                param_dim=param_dim + 1,     # [dt, params] 拼接后作为参数条件
                film_kind="mode",
                film_rank=film_rank,
                use_beta=use_beta,
                gamma_max=gamma_max,
                kernel_kind="cond_kernel",
                kernel_rank=kernel_rank,
                kernel_form=kernel_form,
                channel_low_rank=channel_low_rank,
                kernel_coeff_max=kernel_coeff_max,
                zero_mean_output=zero_mean_output,
            )
        else:
            # M1: 物理空间 channel-wise FiLM(现有行为)
            self.net = FNO2dParam(
                in_channels=in_channels,
                out_channels=state_channels * block_len,
                width=width,
                depth=depth,
                modes=modes,
                time_dim=time_dim,
                param_dim=param_dim + 1,     # [dt, params] 拼接后作为参数条件
                zero_mean_output=zero_mean_output,
            )

    def forward(self, x, flow_time, context, params=None, dt=None,
                forcing=None, geometry_mask=None):
        B, K, C, H, W   = context.shape
        if x is None:
            # 确定性整块回归路径: 未来块无先验,置零输入(与 FNO2dSlot 一致)
            x = torch.zeros(B, self.block_len * C, H, W, device=context.device, dtype=context.dtype)
        if self.cond_mode == "slot":
            # 阶段 3: 显式 slot 轴,整块直通(不做帧通道化展平)
            return self.net(x, flow_time=flow_time, context=context,
                            params=params, dt=dt,
                            forcing=forcing, geometry_mask=geometry_mask)
        context_flat    = context.reshape(B, K * C, H, W)            # §8.1
        extra           = []
        if forcing is not None:
            extra.append(forcing)
        if geometry_mask is not None:
            extra.append(geometry_mask)

        if self.cond_mode == "input_concat":
            # M0: 条件广播拼接,不经过任何调制网络。
            if flow_time is None:
                s_ch = torch.zeros(B, 1, 1, 1, device=x.device, dtype=x.dtype)
            else:
                s_ch = flow_time.reshape(B, 1, 1, 1).to(x.dtype)
            if dt is None:
                dt_ch = torch.zeros(B, 1, 1, 1, device=x.device, dtype=x.dtype)
            else:
                dt_ch = dt.reshape(B, 1, 1, 1).to(x.dtype)
            if params is None:
                p_ch = torch.zeros(B, self.param_dim, 1, 1,
                                   device=x.device, dtype=x.dtype)
            else:
                p_ch = params.reshape(B, -1, 1, 1).to(x.dtype)
            x_input = torch.cat(
                [x, context_flat] + extra
                + [s_ch.expand(B, 1, H, W), dt_ch.expand(B, 1, H, W),
                   p_ch.expand(B, self.param_dim, H, W)],
                dim=1)
            return self.net(x_input)

        x_input         = torch.cat([x, context_flat] + extra, dim=1)

        # [dt, params] 拼接为参数条件(缺省补零,保持 p_mlp 输入形状 P+1 稳定)
        dt_v  = (dt.reshape(B, 1) if dt is not None
                 else torch.zeros(B, 1, device=x.device, dtype=x.dtype))
        p_v   = (params.reshape(B, -1) if params is not None
                 else torch.zeros(B, self.param_dim, device=x.device, dtype=x.dtype))
        parameters = torch.cat([dt_v, p_v], dim=-1)

        if flow_time is not None and flow_time.ndim == 2:
            flow_time = flow_time.squeeze(-1)
        return self.net(x_input, timesteps=flow_time, parameters=parameters)


class UncondFNO2d(nn.Module):
    """无条件 FNO 基线: 整块回归,仅 context 输入,无 s/dt/params 条件。

    forward 签名与 ConditionalFNO2D 完全一致(x, flow_time, context, params, dt, ...),所有条件参数被忽略;
    FM 模式等价于忽略 s 的退化向量场(不建议 FM 使用,基线定位为 det 整块回归)。
    底层复用 FNO2dParam(film_kind="none" + kernel_kind="none" + param_dim=0 -> needs_cond=False,完全无条件的截断谱卷积 FNO)。
    """

    def __init__(self, state_channels: int = 2, context_len: int = 6,
                 block_len: int = 6, param_dim: int = 1,
                 image_size: int = 128, width: int = 64, modes: int = 20,
                 depth: int = 4, time_dim: int = 128,
                 zero_mean_output: bool = False):
        super().__init__()
        self.state_channels = state_channels
        self.context_len    = context_len
        self.block_len      = block_len
        self.param_dim      = param_dim
        self.net            = FNO2dParam(
            in_channels=state_channels * (context_len + block_len),
            out_channels=state_channels * block_len,
            width=width, 
            depth=depth, 
            modes=modes,
            time_dim=time_dim, 
            param_dim=0,
            film_kind="none", 
            kernel_kind="none",
            zero_mean_output=zero_mean_output)

    def forward(self, x, flow_time, context, params=None, dt=None, forcing=None, geometry_mask=None):
        
        B, K, C, H, W   = context.shape
        if x is None:
            x           = torch.zeros(B, self.block_len * C, H, W, device=context.device, dtype=context.dtype)

        context_flat    = context.reshape(B, K * C, H, W)
        extra           = [e for e in (forcing, geometry_mask) if e is not None]
        x_input         = torch.cat([x, context_flat] + extra, dim=1)
        return self.net(x_input)


# =============================================================================
# Slot-mode
# =============================================================================

class SlotMix2d(nn.Module):
    """T_{l,m}(k;c) = T0_{l,m} + Σ_{r=1}^{R_s} a_r(c) A^(r)_{l,m} φ_r(k)

    - T0:  slot basis mixture [S, S],S = K + L.
    - A_r: slot basis [R, S, S];
    - φ_r: mode basis [R, M1+1, M2] (row index = kx, built with the same Hermitian pairing flip as ModeWiseFilm2d);
    - a_r(c) is generated by a conditional projection and zero-initialized -> T = T0 (initially only the base mixing).
    - block_triangular: T[out∈C, in∈X] = 0;
    - mode_basis=False: degenerates to a mode-independent T_{l,m}(c) (ablation control).
    """

    def __init__(self, n_context: int, n_future: int, modes1: int, modes2: int,
                 cond_dim: int, slot_rank: int = 4, mode_basis: bool = True,
                 block_triangular: bool = True, t0_init: str = "identity"):
        super().__init__()
        K, L, S             = n_context, n_future, n_context + n_future
        self.n_context      = n_context
        self.n_future       = n_future
        self.modes1         = modes1
        self.modes2         = modes2
        self.cond_dim       = cond_dim
        self.slot_rank      = slot_rank
        self.mode_basis     = mode_basis

        mask                = torch.ones(S, S)
        if block_triangular:
            mask[:K, K:]    = 0.0                     # T_CX = 0
        self.register_buffer("mask", mask)

        # T_CC = I, T_XX = I, T_XC = 1/K(uniform), T_CX = 0;
        if t0_init == "identity":
            T0              = torch.zeros(S, S)
            T0[:K, :K]      = torch.eye(K)
            T0[K:, K:]      = torch.eye(L)
            T0[K:, :K]      = 1.0 / max(K, 1)
        elif t0_init == "random":
            T0              = 0.1 * torch.randn(S, S)
        else:
            raise ValueError(f"unknow t0_init: {t0_init}")
        self.T0             = nn.Parameter(T0 * mask)

        # low rank
        if slot_rank > 0:
            self.A          = nn.Parameter(0.1 * torch.randn(slot_rank, S, S) * mask)
            if mode_basis:
                # suggest do not use mode_basis, we have proved that it can destroy performance.
                self.phi    = nn.Parameter(0.1 * torch.randn(slot_rank, modes1 + 1, modes2))
            else:
                self.phi    = None
            self.a_proj     = nn.Linear(cond_dim, slot_rank)
            nn.init.zeros_(self.a_proj.weight)
            nn.init.zeros_(self.a_proj.bias)

    @staticmethod
    def _halves(grid):
        """same as ModeWiseFilm2d."""
        M1 = grid.shape[-2] - 1
        return grid[..., :M1, :], torch.flip(grid, dims=[-2])[..., :M1, :]

    def effective_mask_violation(self):
        """Looks at the [matrix actually used in forward]: multiply by mask first, then take forbidden blocks.

        Difference from raw: raw asks "are the params dirty?", this asks
        "does the actual operator leak?". Multiply T0/A by mask (keep allowed
        blocks only) then by forbidden (keep forbidden blocks only); what remains
        is the residue on forbidden blocks of the matrix forward actually uses.
        Normally this should be ≈ 0, meaning the mask truly suppresses the
        forbidden blocks; if > 0, the mask isn't taking effect or is applied
        incorrectly.

        Read the two together:
        - effective ≈ 0            -> operator behavior correct (mask works);
        - raw ≈ 0                  -> params clean (optimizer didn't pollute);
        - raw > 0 but effective≈0  -> output is safe for now, but params are
                                        polluted — keep an eye on it.

        Returns (T0 violation, A violation); A violation is always 0 when
        low-rank is disabled.
        """
        # mask：1 is permit，0 is forbid
        forbidden   = (1.0 - self.mask) # forbidden is the position of mask=0

        # 1. First keep T0's values on allowed blocks and zero out the rest.
        # 2. Also zero out the allowed blocks, leaving only the values on forbidden blocks.
        # 3. t0_v = Max leakage of the base mixing matrix T0 on the forbidden blocks.
        t0_v        = float(((self.T0 * self.mask).abs() * forbidden).max().item())
        # If the low-rank conditional branch is enabled, likewise compute the max leakage of the conditional matrix A on the forbidden blocks
        if self.slot_rank > 0:
            a_v = float(((self.A * self.mask).abs() * forbidden).max().item())
        else:
            a_v = 0.0
        return t0_v, a_v


    def raw_mask_violation(self):
        """Looks at the [raw parameters]: max abs value of T0 / A on forbidden blocks.

        Why check this separately?  In forward, the forbidden blocks are masked
        out *after the fact*, but the optimizer (Adam, etc.) updates the raw
        parameters. So the parameters on forbidden blocks can get "polluted"
        into non-zero values by gradients/momentum. The output is unaffected
        (masked out), but the parameters themselves are no longer clean — once
        the mask logic changes or you do parameter statistics, the problem shows up.

        So this quantity should always be ≈ 0. If clearly > 0, the optimizer has
        polluted the forbidden blocks, meaning you should zero them after each
        optimizer step, or add a regularization constraint.

        Returns (T0 violation, A violation); A violation is always 0 when
        low-rank is disabled.
        """
        forbidden   = (1.0 - self.mask) # forbidden block = where mask is 0
        t0_v        = float((self.T0.abs() * forbidden).max().item())
        a_v         = float((self.A.abs() * forbidden).max().item()) if self.slot_rank > 0 else 0.0
        return t0_v, a_v
    
    def mask_violation(self):
        """Backward-compatible with the old name (equivalent to raw_mask_violation)"""
        return self.raw_mask_violation()

    def forward(self, z_top, z_bot, cond):
        """z_top / z_bot: [B, S, C, M1', M2'] 截断谱块 -> 混合后谱块。

        沿 slot 维混合: q[b,o,c,x,y] = Σ_m T[b,o,m,x,y] z[b,m,c,x,y]。
        块三角约束在 forward 中强制(4.txt §四.3): T0/A 参数可能被优化器
        污染,因此每步都用 self.mask 屏蔽禁止块。
        """
        B, S, C, M1, M2 = z_top.shape
        M1              = min(self.modes1, M1)
        M2              = min(self.modes2, M2)
        T0              = self.T0 * self.mask
        if self.slot_rank > 0 and cond is not None:
            A = self.A * self.mask
            a = self.a_proj(cond)                                   # [B, R]
            if self.mode_basis:
                phi = self.phi[:, :M1 + 1, :M2]
                T = T0[None, :, :, None, None] + torch.einsum(
                    "br, ros, rxy -> bosxy", a, A, phi)             # [B, S, S, M1+1, M2]
                T_top, T_bot = self._halves(T)
                out_top = torch.einsum(
                    "bscxy, bosxy -> bocxy", z_top, T_top.to(z_top.dtype))
                out_bot = torch.einsum(
                    "bscxy, bosxy -> bocxy", z_bot, T_bot.to(z_bot.dtype))
            else:
                # mode-independent slot mixing(5.txt §二: Tc 为 [B,S,S],
                # 不能用五维 einsum,需分支计算)
                Tc = T0[None] + torch.einsum("br, ros -> bos", a, A)  # [B, S, S]
                out_top = torch.einsum(
                    "bscxy, bos -> bocxy", z_top, Tc.to(z_top.dtype))
                out_bot = torch.einsum(
                    "bscxy, bos -> bocxy", z_bot, Tc.to(z_bot.dtype))
        else:
            out_top = torch.einsum("bscxy, os -> bocxy", z_top, T0.to(z_top.dtype))
            out_bot = torch.einsum("bscxy, os -> bocxy", z_bot, T0.to(z_bot.dtype))
        return out_top, out_bot


class SlotFNOBlock(nn.Module):
    """FFT -> SpectralConv2d -> slot-mode mixing T_{l,m}(k;c) -> mode-wise FiLM-> IFFT -> pointwise -> GELU
    """
    def __init__(self, channels: int, modes1: int, modes2: int,
                 cond_dim: int, n_context: int, n_future: int,
                 use_cond_kernel: bool = False, kernel_rank: int = 4,
                 kernel_form: str = "basis", channel_low_rank: int | None = None,
                 kernel_coeff_max: float | None = None,
                 use_mode_film: bool = False, film_rank: int = 8,
                 use_beta: bool = False, gamma_max: float | None = None,
                 slot_rank: int = 4,
                 mode_basis: bool = True, block_triangular: bool = True,
                 t0_init: str = "identity"):
        super().__init__()
        self.channels           = channels
        self.modes1             = modes1
        self.modes2             = modes2
        self.cond_dim           = cond_dim
        self.n_context          = n_context
        self.n_future           = n_future
        self.use_cond_kernel    = use_cond_kernel
        self.use_mode_film      = use_mode_film

        # fft conv
        self.spectral           = SpectralConv2d(channels, channels, modes1, modes2)
        
        # slot-mode mixing
        self.slot_mix           = SlotMix2d(n_context, n_future, modes1, modes2, cond_dim,
                                            slot_rank=slot_rank, mode_basis=mode_basis,
                                            block_triangular=block_triangular, t0_init=t0_init)
        # mode-wise FiLM
        self.mode_film          = ModeWiseFilm2d(channels, modes1, modes2, cond_dim,
                                                 rank=film_rank if use_mode_film else 0, use_beta=use_beta,
                                                 per_slot=n_context + n_future, gamma_max=gamma_max)
        # pointwise + act
        self.pointwise          = nn.Conv2d(channels, channels, kernel_size=1)
        self.act                = nn.GELU()

    def forward(self, h_C, h_X, cond):
        """
        Args:
            h_C (_type_): [B, K, D, H, W]
            h_X (_type_): [B, L, D, H, W]        
            cond (_type_): condition
        """
        B, K, D, H, W   = h_C.shape
        h               = torch.cat([h_C, h_X], dim=1)                  # [B, S, D, H, W]
        S               = h.shape[1]                                    # S = K + L  slots
        W2              = W // 2 + 1
        # fft
        h_ft            = torch.fft.rfft2(h, norm="ortho")              # [B, S, D, H, W2]

        # fftconv
        z               = self.spectral.apply_ft(h_ft.reshape(B * S, D, H, W2))
        z               = z.reshape(B, S, D, H, W2)

        # 
        m1              = min(self.modes1, H // 2)
        m2              = min(self.modes2, W2)
        # slot
        q_top, q_bot    = self.slot_mix(z[:, :, :, :m1, :m2], z[:, :, :, -m1:, :m2], cond)
        # film
        q_top, q_bot    = self.mode_film(q_top, q_bot, cond)
        q_ft            = torch.zeros_like(z)
        q_ft[:, :, :, :m1, :m2]  = q_top
        q_ft[:, :, :, -m1:, :m2] = q_bot

        # ifft
        q               = torch.fft.irfft2(q_ft, s=(H, W), norm="ortho")
        # conv
        p               = self.pointwise(h.reshape(B * S, D, H, W)).reshape(B, S, D, H, W)
        h_out           = self.act(q + p)
        return h_out[:, :K], h_out[:, K:]


class FNO2dSlot(nn.Module):
    """Explicit slot-axis conditional spectral operator.

        - Do 2D FFT on axes (H, W); the time dimension is kept as an explicit slot axis (not a 3D FNO);
        - The context group [B, K, C, H, W] and the future group [B, L, C, H, W] are projected to width via a shared lift, 
            forming a block-triangular slot mixing T = [[T_CC, 0], [T_XC, T_XX]];
        - Per layer: spatial FFT -> spectral kernel -> T_{l,m}(k;c) slot mixing -> mode-wise FiLM (per slot, optional) -> IFFT -> pointwise;
        - The FM interface is fully identical to ConditionalFNO2D:
            forward(x [B, C*L, H, W], flow_time [B], context, params, dt) -> same shape
        - x is the whole block of FM intermediate state, reshaped to [B, L, C, H, W] as the future-group input;
            when x=None, the future group is set to zero (deterministic whole-block regression path).

        Args:
            state_channels: number of state channels C; context_len: number of
                            history frames K; block_len: block length L
            param_dim:      physical parameter dimension P (dt is merged in
                            automatically; p_mlp input is P+1)
            use_cond_kernel: conditional spectral kernel (stage 2, off by default)
            use_mode_film:   mode-wise FiLM (stage 1, off by default)
            slot_rank:       condition rank of slot mixing R_s (0 = T0 only,
                            unconditional mixing)
            mode_basis:      whether slot mixing is mode-dependent
                            (False = T_{l,m}(c))
            block_triangular: block-triangular structure (default True)
            t0_init:         T0 initialization "identity" | "random"
    """
    def __init__(self, state_channels: int = 1, context_len: int = 4,
                 block_len: int = 1, param_dim: int = 1, image_size: int = 64,
                 width: int = 64, modes: int = 20, depth: int = 4,
                 time_dim: int = 128, extra_channels: int = 0,
                 zero_mean_output: bool = False,
                 use_cond_kernel: bool = False, kernel_rank: int = 4,
                 kernel_form: str = "basis", channel_low_rank: int | None = None,
                 kernel_coeff_max: float | None = None,
                 use_mode_film: bool = False, film_rank: int = 8,
                 use_beta: bool = False, gamma_max: float | None = None,
                 slot_rank: int = 4,
                 mode_basis: bool = True, block_triangular: bool = True,
                 t0_init: str = "identity"):
        super().__init__()
        self.state_channels     = state_channels
        self.context_len        = context_len
        self.block_len          = block_len
        self.param_dim          = param_dim
        self.extra_channels     = extra_channels
        self.width              = width
        self.time_dim           = time_dim
        self.zero_mean_output   = zero_mean_output
        self.use_cond_kernel    = use_cond_kernel
        self.use_mode_film      = use_mode_film

        # Conditional encoding (same convention as ConditionalFNO2D: s -> t_emb, [dt, params] -> p_emb)
        self.t_mlp              = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.p_mlp              = nn.Sequential(nn.Linear(param_dim + 1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))

        # shared lift: state C + extra cond + grid(2)
        self.lift               = nn.Conv2d(state_channels + extra_channels + 2, width, kernel_size=1)
        cond_dim                = 2 * time_dim                      # [t_emb, p_emb] concat
        self.blocks             = nn.ModuleList([
            SlotFNOBlock(width, modes, modes, cond_dim, context_len, block_len,
                         use_cond_kernel=use_cond_kernel, kernel_rank=kernel_rank,
                         kernel_form=kernel_form, channel_low_rank=channel_low_rank,
                         kernel_coeff_max=kernel_coeff_max,
                         use_mode_film=use_mode_film, film_rank=film_rank,
                         use_beta=use_beta, gamma_max=gamma_max, slot_rank=slot_rank,
                         mode_basis=mode_basis, block_triangular=block_triangular,
                         t0_init=t0_init)
            for _ in range(depth)
        ])

        self.proj = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, state_channels, kernel_size=1),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    @staticmethod
    def _grid(B: int, H: int, W: int, device, dtype):
        gx      = torch.arange(H, device=device, dtype=dtype) / H
        gy      = torch.arange(W, device=device, dtype=dtype) / W
        gx, gy  = torch.meshgrid(gx, gy, indexing='ij')
        grid    = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return grid

    def forward(self, x, flow_time, context, params=None, dt=None, forcing=None, geometry_mask=None):
        B, K, C, H, W   = context.shape
        L               = self.block_len
        if x is None:
            x_slots     = torch.zeros(B, L, C, H, W, device=context.device, dtype=context.dtype)
        else:
            x_slots     = x.reshape(B, L, C, H, W)

        grid            = self._grid(B, H, W, context.device, context.dtype)
        extras          = [e for e in (forcing, geometry_mask) if e is not None]
        ctx_in          = torch.cat(
            [context.reshape(B * K, C, H, W), grid.repeat_interleave(K, 0)]+ [e.repeat_interleave(K, 0) for e in extras], dim=1)
        
        fut_in          = torch.cat(
            [x_slots.reshape(B * L, C, H, W), grid.repeat_interleave(L, 0)]+ [e.repeat_interleave(L, 0) for e in extras], dim=1)
        
        assert ctx_in.shape[1] == self.lift.in_channels, f"ctx_in channel {ctx_in.shape[1]} != lift.in_channels {self.lift.in_channels}"
        assert fut_in.shape[1] == self.lift.in_channels, f"fut_in channel {fut_in.shape[1]} != lift.in_channels {self.lift.in_channels}"

        h_C             = self.lift(ctx_in).reshape(B, K, self.width, H, W)
        h_X             = self.lift(fut_in).reshape(B, L, self.width, H, W)

        # condition: s -> t_emb;[dt, params] -> p_emb(zero-padded by default to avoid shape mismatch)
        if flow_time is None:
            t_emb       = torch.zeros(B, self.time_dim, device=context.device, dtype=context.dtype)
        else:
            if flow_time.ndim == 2:
                flow_time = flow_time.squeeze(-1)
            t_emb       = self.t_mlp(_sinusoidal_time_embedding(flow_time, self.time_dim))
        dt_v            = dt.reshape(B, 1) if dt is not None else torch.zeros(B, 1, device=context.device, dtype=context.dtype)
        p_v             = params.reshape(B, -1) if params is not None else torch.zeros(B, self.param_dim, device=context.device, dtype=context.dtype)
        p_emb           = self.p_mlp(torch.cat([dt_v, p_v], dim=-1))
        cond            = torch.cat([t_emb, p_emb], dim=-1)       # [B, 2*time_dim]

        for blk in self.blocks:
            h_C, h_X    = blk(h_C, h_X, cond)

        out             = self.proj(h_X.reshape(B * L, self.width, H, W)).reshape(B, L, C, H, W)
        out             = out.reshape(B, C * L, H, W)
        if self.zero_mean_output:
            out         = out - out.mean(dim=(-2, -1), keepdim=True)
        return out


# =============================================================================
#   module clusters
# =============================================================================

class FNO2dInputConcat(ConditionalFNO2D):
    """M0 baseline: FNO-InputConcat 
        — conditions (s, dt, params) are broadcast and concatenated, with no modulation.

        u'(x) = [x_s, u_{t-K+1:t}(x), s, dt, params](x)
        The underlying net is built with film_kind="none": the FNO block is a purely unconditional spectral layer.
    """
    def __init__(self, **kwargs):
        kwargs["cond_mode"] = "input_concat"
        super().__init__(**kwargs)


class FNO2dGlobalFilm(ConditionalFNO2D):
    """M1 baseline: FNO-GlobalFiLM —— spatial channel-wise FiLM.

    (s, dt, params) is embedded by FiLM into each FNO block.
    """

    def __init__(self, **kwargs):
        kwargs["cond_mode"] = "global_film"
        super().__init__(**kwargs)


class FNO2dModeFilm(ConditionalFNO2D):
    """M2 baseline: FNO-ModeFiLM —— frequent mode-wise FiLM.

        ŷ_o(k) = (1 + γ_o(k;c)) ⊙ (W_0(k) ⊛ x̂(k))_o [+ β_o(k;c)]

        s is timesteps, [dt, params] isparameters.
    """

    def __init__(self, **kwargs):
        kwargs["cond_mode"] = "mode_film"
        super().__init__(**kwargs)



class FNO2dSlotModeFilm(ConditionalFNO2D):
    """Explicit slot + mode-wise FiLM
    """

    def __init__(self, **kwargs):
        kwargs["cond_mode"] = "slot"
        kwargs.setdefault("use_cond_kernel", False)
        kwargs.setdefault("use_mode_film", True)
        super().__init__(**kwargs)


# size module
COND_FNO_CONFIGS = {
    "s": dict(width=32,  modes=12, depth=4, time_dim=128),
    "b": dict(width=64,  modes=20, depth=4, time_dim=128),
    "l": dict(width=96,  modes=24, depth=6, time_dim=128),
}


def build_cond_fno(cond_mode: str = "global_film", size: str = "b",
                   state_channels: int = 1, context_len: int = 4,
                   block_len: int = 1, param_dim: int = 1,
                   image_size: int = 64, **kwargs):
    """make condition FNO clusters

    Args:
        cond_mode: "input_concat"(M0)| "global_film"(M1)| "mode_film"(M2)| "slot"(explicit slot baseline)|"slot_film"(G)
        size:      s / b / l module(COND_FNO_CONFIGS)
    """
    assert cond_mode in ("input_concat", "global_film", "mode_film",
                         "cond_kernel", "mode_film_kernel",
                         "slot", "slot_kernel",
                         "slot_full", "slot_film"), f"未知 cond_mode: {cond_mode}"
    assert size in ("s", "b", "l"), f"未知 size: {size}"
    cfg = dict(COND_FNO_CONFIGS[size])
    cfg.update(kwargs)
    cls = {"input_concat": FNO2dInputConcat,
           "global_film":  FNO2dGlobalFilm,
           "mode_film":    FNO2dModeFilm,
           "slot":         FNO2dSlot,
           "slot_film":    FNO2dSlotModeFilm}[cond_mode]
    return cls(state_channels=state_channels, context_len=context_len,
               block_len=block_len, param_dim=param_dim,
               image_size=image_size, **cfg)