"""
2D Fourier Neural Operator for periodic Kolmogorov-flow super-resolution.

References:
    Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A.,
    Anandkumar, A. "Fourier Neural Operator for Parametric Partial Differential
    Equations." ICLR 2021.

Why FNO for our task:
    - Kolmogorov flow lives on a 2*pi periodic torus; FNO's spectral convolution
      is exactly periodic by construction (no patch boundary artefacts that
      hurt the divergence in DiT).
    - Energy spectrum of forced 2D turbulence concentrates at low/medium wavenumbers,
      which matches FNO's truncated-mode design.
    - Spectral mixing is global with O(N log N) cost; gives long-range receptive
      field without depth, beneficial for super-resolution at 128x128.
    - Truncated modes act as a low-pass filter -> output is smooth, derivatives
      (used in physics_layer for hard constraint) are numerically stable.

Two variants are exposed:
    - FNO2d_SR    : regression model, outputs 2 channels [u, v].
    - FNO2d_HARD  : hard-constraint model, outputs 3 channels [psi, u_mean, v_mean]
                    designed to be wrapped by `physics_layer.PhysicsInformedSuperResModel`.

Forward signature is kept compatible with the rest of the project:
    model(x, timesteps=None, y=None)
where:
    x         : high-res tensor [B, in_channels, H, W] (or None for pure regression)
    timesteps : optional flow-matching/diffusion timestep, embedded via sin/cos MLP
    y         : low-res conditioning [B, in_channels, H/scale, W/scale]
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    Truncated 2D spectral convolution. Multiplies the lowest (modes1, modes2)
    Fourier coefficients with a learnable complex weight. Periodic by construction.
    """
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1       = modes1
        self.modes2       = modes2

        scale = 1.0 / (in_channels * out_channels)
        # We split complex weights into real and imaginary parts to support DDP/NCCL
        self.weights1_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2)
        )
        self.weights1_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2)
        )
        self.weights2_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2)
        )
        self.weights2_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2)
        )

    @property
    def weights1(self):
        return torch.complex(self.weights1_real, self.weights1_imag)

    @property
    def weights2(self):
        return torch.complex(self.weights2_real, self.weights2_imag)

    @staticmethod
    def _compl_mul2d(x, w):
        # x: (B, in_c, M1, M2), w: (in_c, out_c, M1, M2) -> (B, out_c, M1, M2)
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def apply_ft(self, x_ft, m1=None, m2=None):
        """截断频域乘法(不含 FFT): x_ft [B, in, H, W//2+1] -> [B, out, H, W//2+1]。

        只填充保留 mode 的两块(rfft 布局的上半块 kx>=0 与下半块 kx<0),
        其余频点置零 —— 与 forward 语义完全一致,供需要频域中间结果的
        下游模块(如 Mode-wise FiLM)复用。显式 m1/m2 统一 clamp(6.txt §三.2)。
        """
        B, _, H, W2 = x_ft.shape
        m1 = min(self.modes1, H // 2, m1 if m1 is not None else self.modes1)
        m2 = min(self.modes2, W2,     m2 if m2 is not None else self.modes2)
        out_ft = torch.zeros(B, self.out_channels, H, W2,
                             dtype=torch.cfloat, device=x_ft.device)
        out_ft[:, :, :m1, :m2] = self._compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self._compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )
        return out_ft

    def forward(self, x):
        B, _, H, W = x.shape
        out_ft = self.apply_ft(torch.fft.rfft2(x, norm="ortho"))
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOBlock(nn.Module):
    """
    Standard FNO block: spectral_conv + pointwise(1x1 conv) + GELU,
    optionally with FiLM-style modulation by a timestep embedding and/or
    a physical-parameter embedding.
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
        h = self.spectral(x) + self.pointwise(x)
        if self.cond_dim > 0:
            cond_list = [e for e in (t_emb, p_emb) if e is not None]
            if cond_list:
                cond = torch.cat(cond_list, dim=-1)
                scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
                scale = scale.unsqueeze(-1).unsqueeze(-1)
                shift = shift.unsqueeze(-1).unsqueeze(-1)
                h = h * (1 + scale) + shift
        return self.act(h)


# =============================================================================
# 阶段 1: Mode-wise FiLM（14.txt M2 / 13.txt §7.2）
# =============================================================================

class ModeWiseFilm2d(nn.Module):
    """Mode-wise FiLM: 条件依赖的 mode-wise 对角谱调制（阶段 1）。

        ŷ_o(k) = (1 + γ_o(k;c)) ⊙ ẑ_o(k) [+ β_o(k;c)]

    要点:
      - γ 为实数,作用于 rfft2 存储半谱的两个 kx 半块（上块 kx>=0 / 下块 kx<0,
        下块行 u 与上块行 M1-u 配对）。输出经 irfft2 恒为实值;
      - **受限 mode basis（4.txt §二.3 修正）**:增益定义在存储半谱上,不保证
        γ(-k)=γ(k)(非径向/偶对称),因此是「受限 mode gain 算子」,而非严格的
        k-偶对称物理算子——论文中应按此表述,不使用「完全保证 Hermitian
        一致性」的强措辞;
      - 低秩生成(13.txt §7.2 推荐): γ_o(k;c) = Σ_r a_{o,r}(c) φ_r(k),
        φ_r ∈ R^{R x (M1+1) x M2} 为可学习 mode 基（行索引 = kx ∈ [0, M1],
        含与下半块配对的 kx=M1 行）,a 由条件投影生成;
      - 零初始化 a / β 投影 -> 初始恒等调制(退化回无条件 FNO,稳定训练);
      - β 默认关闭(13.txt §7.2: 第一版只使用乘性 gain)。
    输入 z_top / z_bot 可带 G 维(组/slot 维),为阶段 3 slot 模型复用预留。
    """

    def __init__(self, channels: int, modes1: int, modes2: int,
                 cond_dim: int, rank: int = 8, use_beta: bool = False,
                 per_slot: int = 1, gamma_max: float | None = None):
        super().__init__()
        self.channels = channels
        self.modes1   = modes1
        self.modes2   = modes2
        self.cond_dim = cond_dim
        self.rank     = rank
        self.use_beta = use_beta
        self.per_slot = per_slot
        assert gamma_max is None or gamma_max > 0, \
            f"gamma_max 必须 > 0(6.txt §三.3),got {gamma_max}"
        self.gamma_max = gamma_max     # 有界增益: γ = γ_max·tanh(γ̃)(5.txt §七);None=无界

        # 可学习 mode 基 φ_r(k): [R, M1+1, M2],行索引 = kx ∈ [0, M1]
        # rank=0 时关闭调制(恒等),供仅条件化谱核的 block 复用。
        self.phi = (nn.Parameter(0.1 * torch.randn(rank, modes1 + 1, modes2))
                    if rank > 0 else None)
        # per_slot: 逐 slot 系数数量(G 维,阶段 3 slot 模型);默认 1 = 无 slot 轴
        self.a_proj = (nn.Linear(cond_dim, per_slot * channels * rank)
                       if rank > 0 else None)
        if self.a_proj is not None:
            nn.init.zeros_(self.a_proj.weight)
            nn.init.zeros_(self.a_proj.bias)
        self.b_proj = None
        if use_beta and rank > 0:
            self.b_proj = nn.Linear(cond_dim, per_slot * channels * rank)
            nn.init.zeros_(self.b_proj.weight)
            nn.init.zeros_(self.b_proj.bias)

    @staticmethod
    def _gain_halves(grid):
        """由 kx ∈ [0, M1] 增益网格构造上下半块增益（kx 行翻转配对）。

        grid: [B, G, C, M1+1, M2],行索引 = kx。
        下块行 u ∈ [0, M1) 对应 kx = u - M1(<0;u=0 且 M1=H/2 时为 Nyquist 行),
        与上块行 j = M1 - u 配对 -> 下块增益 = 行翻转(grid) 的前 M1 行。
        注意:仅按 kx 配对,ky 取存储半谱同索引——受限 mode gain(4.txt §二.3)。
        """
        M1 = grid.shape[-2] - 1
        return grid[:, :, :, :M1, :], torch.flip(grid, dims=[-2])[:, :, :, :M1, :]

    def forward(self, z_top, z_bot, cond):
        """z_top / z_bot: [B, C, M1', M2'] 或 [B, G, C, M1', M2'] 截断谱块。

        返回同形状的调制后谱块(M1' <= modes1, M2' <= modes2,按输入截断)。
        rank=0 时返回原样(恒等调制)。
        """
        if self.rank == 0:
            return z_top, z_bot
        is4d = z_top.ndim == 4
        if is4d:
            z_top, z_bot = z_top.unsqueeze(1), z_bot.unsqueeze(1)
        B, G, C, M1, M2 = z_top.shape
        assert G == self.per_slot, \
            f"输入 G={G} 与构造 per_slot={self.per_slot} 不一致"
        M1 = min(self.modes1, M1)
        M2 = min(self.modes2, M2)
        phi = self.phi[:, :M1 + 1, :M2]

        a     = self.a_proj(cond).reshape(B, G, C, self.rank)
        grid  = torch.einsum("bgcr, rxy -> bgcxy", a, phi)  # [B, G, C, M1+1, M2]
        if self.gamma_max is not None:
            grid = self.gamma_max * torch.tanh(grid)        # 有界增益(5.txt §七)
        g_top, g_bot = self._gain_halves(grid)
        z_top = z_top * (1 + g_top)
        z_bot = z_bot * (1 + g_bot)
        if self.use_beta:
            b     = self.b_proj(cond).reshape(B, G, C, self.rank)
            bg    = torch.einsum("bgcr, rxy -> bgcxy", b, phi)
            bt, bb = self._gain_halves(bg)
            z_top = z_top + bt
            z_bot = z_bot + bb
        if is4d:
            z_top, z_bot = z_top.squeeze(1), z_bot.squeeze(1)
        return z_top, z_bot


class FNOBlockModeFilm(nn.Module):
    """FNO block + Mode-wise FiLM（频域,谱层输出上调制），可选条件化谱核。

        结构: h = GELU( irfft2( 调制( W(k;c) ⊛ x̂(k) ) ) + W_1(x) )

    与 FNOBlock 相同的条件约定: t_emb / p_emb 拼接为 cond。所有条件路径
    零初始化 -> 初始等价于无条件 FNO block。

    Args:
        film_rank:      mode-wise FiLM 的 mode 基数量;0 = 关闭(仅保留谱核条件化)
        kernel_rank:    条件化谱核基数量;0 = 固定谱核 W_0（默认）
        kernel_form / channel_low_rank: 见 ConditionalSpectralConv2d
    """

    def __init__(self, channels: int, modes1: int, modes2: int,
                 time_dim: int = 0, param_dim: int = 0,
                 film_rank: int = 8, use_beta: bool = False,
                 kernel_rank: int = 0, kernel_form: str = "basis",
                 channel_low_rank: int | None = None,
                 gamma_max: float | None = None,
                 kernel_coeff_max: float | None = None):
        super().__init__()
        self.channels  = channels
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.act       = nn.GELU()
        self.time_dim  = time_dim
        self.param_dim = param_dim
        self.kernel_rank = kernel_rank
        # t_emb 与 p_emb 均为 time_dim 维,拼接后为 cond。
        self.cond_dim  = time_dim + (time_dim if param_dim > 0 else 0)
        if kernel_rank > 0:
            # 阶段 2: 条件化谱核 W(k;c) = W_0(k) + Σ_r a_r(c) W_r(k)
            self.spectral = ConditionalSpectralConv2d(
                channels, channels, modes1, modes2, self.cond_dim,
                basis_rank=kernel_rank, kernel_form=kernel_form,
                channel_low_rank=channel_low_rank,
                kernel_coeff_max=kernel_coeff_max)
        else:
            self.spectral = SpectralConv2d(channels, channels, modes1, modes2)
        # 阶段 1: mode-wise FiLM(film_rank=0 时为恒等)
        self.mode_film = ModeWiseFilm2d(channels, modes1, modes2, self.cond_dim,
                                        rank=film_rank, use_beta=use_beta,
                                        gamma_max=gamma_max)

    def forward(self, x, t_emb=None, p_emb=None):
        B, _, H, W = x.shape
        cond_list = [e for e in (t_emb, p_emb) if e is not None]
        if cond_list:
            cond = torch.cat(cond_list, dim=-1)
        else:
            cond = torch.zeros(B, self.cond_dim, device=x.device, dtype=x.dtype)

        x_ft   = torch.fft.rfft2(x, norm="ortho")               # [B, C, H, W//2+1]
        if self.kernel_rank > 0:
            out_ft = self.spectral.apply_ft(x_ft, cond)         # 条件化谱卷积
        else:
            out_ft = self.spectral.apply_ft(x_ft)               # 截断谱卷积
        m1     = min(self.spectral.modes1, H // 2)
        m2     = min(self.spectral.modes2, W // 2 + 1)
        top, bot = self.mode_film(out_ft[:, :, :m1, :m2],
                                  out_ft[:, :, -m1:, :m2], cond)
        # 用新张量组装(勿写回 out_ft: top/bot 是其视图,原地写会破坏反传图)
        q_ft   = torch.zeros_like(out_ft)
        q_ft[:, :, :m1, :m2]   = top
        q_ft[:, :, -m1:, :m2]  = bot
        q = torch.fft.irfft2(q_ft, s=(H, W), norm="ortho")
        return self.act(q + self.pointwise(x))


# =============================================================================
# 阶段 2: 条件化谱核（14.txt M3 / 13.txt §7.4）
# =============================================================================

class ConditionalSpectralConv2d(nn.Module):
    """条件化谱核（阶段 2）: W_eff(k;c) = W_0(k) + Σ_{r=1}^{R} a_r(c) W_r(k)。

    - W_0 与 SpectralConv2d 同约定: 复权重拆实/虚部(兼容 DDP/NCCL),上下
      两个 Hermitian 半块(rfft 布局 kx>=0 / kx<0)各一套权重;
    - W_r 为可学习谱核基;条件系数 a_r = a_proj(cond) 零初始化 ->
      W_eff = W_0,初始严格退化为无条件谱卷积(稳定训练,14.txt §四);
    - kernel_form="basis"（默认）: 逐 r 施加基核后加权求和
      （内存友好,不物化 batch 依赖核;13.txt §7.4 的循环实现）;
    - kernel_form="channel_low_rank": W_r = U_r V_r^H（q=channel_low_rank),
      两步 einsum 降低参数量与显存（13.txt §7.4 推荐的内存优化形式）。

    forward 签名与 SpectralConv2d 对齐:
        apply_ft(x_ft, cond=None) : rfft 布局截断频域乘法
        forward(x, cond=None)      : 完整 谱卷积 + irfft2
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int, cond_dim: int,
                 basis_rank: int = 4, kernel_form: str = "basis",
                 channel_low_rank: int | None = None,
                 kernel_coeff_max: float | None = None):
        super().__init__()
        assert kernel_form in ("basis", "channel_low_rank"), \
            f"未知 kernel_form: {kernel_form}"
        assert kernel_coeff_max is None or kernel_coeff_max > 0, \
            f"kernel_coeff_max 必须 > 0(21.txt §一),got {kernel_coeff_max}"
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1       = modes1
        self.modes2       = modes2
        self.cond_dim     = cond_dim
        self.basis_rank   = basis_rank
        self.kernel_form  = kernel_form
        self.channel_low_rank = channel_low_rank
        self.kernel_coeff_max = kernel_coeff_max
        # a_r(c) = kernel_coeff_max * tanh(ã_r(c)); None = unbounded(21.txt §一)

        scale = 1.0 / (in_channels * out_channels)
        # W_0: 基础谱核(与 SpectralConv2d 同约定)
        self.w0_1_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.w0_1_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.w0_2_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.w0_2_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))

        if basis_rank > 0:
            bscale = scale / math.sqrt(max(basis_rank, 1))
            if kernel_form == "basis":
                # W_r: [R, in, out, M1, M2] 完整基核
                self.basis1_real = nn.Parameter(bscale * torch.randn(basis_rank, in_channels, out_channels, modes1, modes2))
                self.basis1_imag = nn.Parameter(bscale * torch.randn(basis_rank, in_channels, out_channels, modes1, modes2))
                self.basis2_real = nn.Parameter(bscale * torch.randn(basis_rank, in_channels, out_channels, modes1, modes2))
                self.basis2_imag = nn.Parameter(bscale * torch.randn(basis_rank, in_channels, out_channels, modes1, modes2))
            else:
                # W_r = U_r V_r^H: U [R, in, q, M1, M2], V [R, out, q, M1, M2]
                q = channel_low_rank or max(in_channels // 4, 1)
                self.u1_real = nn.Parameter(bscale * torch.randn(basis_rank, in_channels, q, modes1, modes2))
                self.u1_imag = nn.Parameter(bscale * torch.randn(basis_rank, in_channels, q, modes1, modes2))
                self.v1_real = nn.Parameter(bscale * torch.randn(basis_rank, out_channels, q, modes1, modes2))
                self.v1_imag = nn.Parameter(bscale * torch.randn(basis_rank, out_channels, q, modes1, modes2))
                self.u2_real = nn.Parameter(bscale * torch.randn(basis_rank, in_channels, q, modes1, modes2))
                self.u2_imag = nn.Parameter(bscale * torch.randn(basis_rank, in_channels, q, modes1, modes2))
                self.v2_real = nn.Parameter(bscale * torch.randn(basis_rank, out_channels, q, modes1, modes2))
                self.v2_imag = nn.Parameter(bscale * torch.randn(basis_rank, out_channels, q, modes1, modes2))
                self.channel_low_rank = q
                # 不再使用UV校准，对结果影响不大
                # 初始化尺度校准(7.txt §七.3/§八): 旧尺度下
                # RMS(U_rV_r^H) ≈ √q·(bscale√2)² << RMS(W_r^basis) = bscale√2,
                # 导致 lowrank 条件核欠激活(ΔW/W₀ 仅 0.04-0.18 vs basis 0.72-0.95)。
                # 逐 r 缩放 U_r 与 V_r,使外积 RMS 对齐 basis 目标尺度。
                # with torch.no_grad():
                #     target_rms = bscale * math.sqrt(2.0)
                #     for (u_re, u_im, v_re, v_im) in (
                #             (self.u1_real, self.u1_imag, self.v1_real, self.v1_imag),
                #             (self.u2_real, self.u2_imag, self.v2_real, self.v2_imag)):
                #         for r in range(basis_rank):
                #             W_r = torch.einsum(
                #                 "iqxy,oqxy->ioxy",
                #                 torch.complex(u_re[r], u_im[r]),
                #                 torch.complex(v_re[r], v_im[r]).conj())
                #             rms_r = float(W_r.abs().square().mean().sqrt())
                #             s_r = math.sqrt(target_rms / max(rms_r, 1e-12))
                #             s_r = min(max(s_r, 0.01), 100.0)
                #             for p in (u_re, u_im, v_re, v_im):
                #                 p.data[r] = p.data[r] * s_r
            # 条件系数 a_r(c),零初始化 -> 初始退化为 W_0
            self.a_proj = nn.Linear(cond_dim, basis_rank)
            nn.init.zeros_(self.a_proj.weight)
            nn.init.zeros_(self.a_proj.bias)
        else:
            self.a_proj = None

    @property
    def w0_1(self):
        return torch.complex(self.w0_1_real, self.w0_1_imag)

    @property
    def w0_2(self):
        return torch.complex(self.w0_2_real, self.w0_2_imag)

    @property
    def basis1(self):
        return torch.complex(self.basis1_real, self.basis1_imag)

    @property
    def basis2(self):
        return torch.complex(self.basis2_real, self.basis2_imag)

    @property
    def u1(self):
        return torch.complex(self.u1_real, self.u1_imag)

    @property
    def v1(self):
        return torch.complex(self.v1_real, self.v1_imag)

    @property
    def u2(self):
        return torch.complex(self.u2_real, self.u2_imag)

    @property
    def v2(self):
        return torch.complex(self.v2_real, self.v2_imag)

    @staticmethod
    def _compl_mul2d(x, w):
        # x: (B, in_c, M1, M2), w: (in_c, out_c, M1, M2) -> (B, out_c, M1, M2)
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def _kernel_apply(self, x_top, x_bot, a):
        """条件残差 ΔW(k;c) ⊛ x̂,返回 (delta_top, delta_bot)。

        基核/低秩因子按输入实际截断 (m1, m2) 切片(与 apply_ft 中基权重
        weights1/2 的截断语义一致;分辨率 < 模板 modes 时避免尺寸错配)。
        """
        B = x_top.shape[0]
        m1 = x_top.shape[-2]
        m2 = x_top.shape[-1]
        if self.kernel_form == "basis":
            delta_top = torch.zeros_like(x_top)
            delta_bot = torch.zeros_like(x_bot)
            for r in range(self.basis_rank):
                ar = a[:, r].view(B, 1, 1, 1)
                delta_top = delta_top + ar * self._compl_mul2d(
                    x_top, self.basis1[r][:, :, :m1, :m2])
                delta_bot = delta_bot + ar * self._compl_mul2d(
                    x_bot, self.basis2[r][:, :, :m1, :m2])
            return delta_top, delta_bot
        # channel_low_rank: ΔW = Σ_r a_r U_r V_r^H,先逐 r 投影再按 a_r 加权
        # (修复 4.txt §三.2:旧实现把 a 同时乘到 U/V 两侧,产生 a_r^2 与交叉项)。
        u1, v1 = self.u1[:, :, :, :m1, :m2], self.v1[:, :, :, :m1, :m2]
        u2, v2 = self.u2[:, :, :, :m1, :m2], self.v2[:, :, :, :m1, :m2]
        a_c = a.to(self.u1.dtype)                       # [B, R] 复数系数(仅加权用)
        ux = torch.einsum("bixy, riqxy -> brqxy", x_top, u1)
        delta_r = torch.einsum("brqxy, roqxy -> broxy",
                               ux, torch.conj(v1))
        delta_top = torch.einsum("br, broxy -> boxy", a_c, delta_r)
        ux = torch.einsum("bixy, riqxy -> brqxy", x_bot, u2)
        delta_r = torch.einsum("brqxy, roqxy -> broxy",
                               ux, torch.conj(v2))
        delta_bot = torch.einsum("br, broxy -> boxy", a_c, delta_r)
        return delta_top, delta_bot

    def apply_ft(self, x_ft, cond=None, m1=None, m2=None):
        """截断频域乘法(不含 FFT): x_ft [B, in, H, W//2+1] -> [B, out, H, W//2+1]。

        与 SpectralConv2d.apply_ft 语义一致;cond 给出时叠加条件残差核。
        """
        B, _, H, W2 = x_ft.shape
        m1 = min(self.modes1, H // 2, m1 if m1 is not None else self.modes1)
        m2 = min(self.modes2, W2,     m2 if m2 is not None else self.modes2)
        x_top = x_ft[:, :, :m1, :m2]
        x_bot = x_ft[:, :, -m1:, :m2]
        out_top = self._compl_mul2d(x_top, self.w0_1[:, :, :m1, :m2])
        out_bot = self._compl_mul2d(x_bot, self.w0_2[:, :, :m1, :m2])
        if self.a_proj is not None and cond is not None:
            a = self.a_proj(cond)                       # [B, R]
            if self.kernel_coeff_max is not None:
                a = self.kernel_coeff_max * torch.tanh(a)  # 系数有界门控(21.txt §一)
            d_top, d_bot = self._kernel_apply(x_top, x_bot, a)
            out_top = out_top + d_top
            out_bot = out_bot + d_bot
        out_ft = torch.zeros(B, self.out_channels, H, W2,
                             dtype=torch.cfloat, device=x_ft.device)
        out_ft[:, :, :m1, :m2]  = out_top
        out_ft[:, :, -m1:, :m2] = out_bot
        return out_ft

    def forward(self, x, cond=None):
        B, _, H, W = x.shape
        out_ft = self.apply_ft(torch.fft.rfft2(x, norm="ortho"), cond)
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
    half    = dim // 2
    freqs   = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args    = t.float()[:, None] * freqs[None]
    emb     = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class FNO2d(nn.Module):
    """
    Backbone:
        Lift (1x1 conv): in_channels   -> width
        N x FNOBlock  : width          -> width
        Project       : width          -> 128 -> out_channels (two 1x1 convs + GELU)

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
        self.in_channels      = in_channels
        self.cond_in_channels = cond_in_channels
        self.out_channels     = out_channels
        self.use_lr_shortcut  = use_lr_shortcut
        self.time_dim         = time_dim

        # Time/timestep embedding (used only when timesteps is not None).
        self.t_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Lift: x and upsampled y are concatenated -> 2 * cond_in_channels channels (or in+cond).
        lift_in = in_channels + cond_in_channels + 2  # +2 for grid (x, y) coords
        self.lift = nn.Conv2d(lift_in, width, kernel_size=1)

        self.blocks = nn.ModuleList([
            FNOBlock(width, modes, modes, time_dim=time_dim) for _ in range(depth)
        ])

        self.proj = nn.Sequential(
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
        gx = torch.arange(H, device=device, dtype=dtype) / H
        gy = torch.arange(W, device=device, dtype=dtype) / W
        gx, gy = torch.meshgrid(gx, gy, indexing='ij')
        grid = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
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
    Parameter-conditioned FNO vector field for Flow Matching on
    divergence-free stream-function (psi) states.

    Difference from FNO2d (super-resolution):
      - No low-res conditioning `y`; conditioning is a low-dimensional
        physical-parameter vector `parameters` (e.g. Re), embedded by an MLP
        and injected into every FNO block via FiLM.
      - Output is the FM vector field in psi-space, forced to zero-mean so the
        ODE trajectory stays in the feasible (gauge-fixed) tangent space.
      - SpectralConv2d's truncated modes act as an intrinsic low-pass filter,
        which suppresses the high-frequency noise amplification that a plain
        CNN suffers from when decoding psi -> velocity.

    Forward: (x, timesteps=None, parameters=None) -> dx  [B, out_channels, H, W]
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
        kernel_rank: int = 4,
        kernel_form: str = "basis",
        channel_low_rank: int | None = None,
        kernel_coeff_max: float | None = None,
    ):
        super().__init__()
        assert film_kind in ("none", "global", "mode"), \
            f"未知 film_kind: {film_kind}"
        assert kernel_kind in ("none", "cond_kernel"), \
            f"未知 kernel_kind: {kernel_kind}"
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.width        = width
        self.depth        = depth
        self.modes        = modes
        self.time_dim     = time_dim
        self.param_dim    = param_dim
        self.zero_mean_output = zero_mean_output
        self.film_kind    = film_kind
        self.film_rank    = film_rank
        self.use_beta     = use_beta
        self.gamma_max    = gamma_max
        self.kernel_kind  = kernel_kind
        self.kernel_rank  = kernel_rank

        # 条件编码分支: 只要有任一调制/核条件化就需要 t_mlp / p_mlp;
        # film_kind="none" 且 kernel_kind="none" 时条件改由输入通道拼接注入
        # (对应 M0 FNO-InputConcat 基线)。
        needs_cond = film_kind != "none" or kernel_kind == "cond_kernel"
        self.needs_cond = needs_cond
        self.t_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        ) if needs_cond else None

        # Physical-parameter embedding (e.g. normalized Reynolds number).
        if param_dim > 0 and needs_cond:
            self.p_mlp = nn.Sequential(
                nn.Linear(param_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            self.p_mlp = None

        # Lift: x + grid coordinates (no low-res conditioning).
        lift_in = in_channels + 2
        self.lift = nn.Conv2d(lift_in, width, kernel_size=1)

        # 条件注入方式（film_kind x kernel_kind）:
        #   kernel_kind="cond_kernel" -> 条件化谱核(阶段 2),可叠加 mode film
        #     (film_kind="mode",即消融 D);film_kind 否则关闭;
        #   film_kind="mode"          -> 频域 mode-wise FiLM(M2,阶段 1);
        #   film_kind="global"        -> 物理空间 channel-wise FiLM(M1);
        #   其余                      -> 无调制(M0,条件走输入拼接)。
        if kernel_kind == "cond_kernel":
            block = lambda: FNOBlockModeFilm(
                width, modes, modes, time_dim=time_dim, param_dim=param_dim,
                film_rank=film_rank if film_kind == "mode" else 0,
                use_beta=use_beta,
                kernel_rank=kernel_rank, kernel_form=kernel_form,
                channel_low_rank=channel_low_rank, gamma_max=gamma_max,
                kernel_coeff_max=kernel_coeff_max)
        elif film_kind == "mode":
            block = lambda: FNOBlockModeFilm(
                width, modes, modes, time_dim=time_dim, param_dim=param_dim,
                film_rank=film_rank, use_beta=use_beta, gamma_max=gamma_max)
        elif film_kind == "global":
            block = lambda: FNOBlock(
                width, modes, modes, time_dim=time_dim, param_dim=param_dim)
        else:
            block = lambda: FNOBlock(width, modes, modes, time_dim=0, param_dim=0)
        self.blocks = nn.ModuleList([block() for _ in range(depth)])

        self.proj = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=1),
        )

        # Zero-init the final projection: the flow starts from the zero field.
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    @staticmethod
    def _grid(B: int, H: int, W: int, device, dtype):
        gx = torch.arange(H, device=device, dtype=dtype) / H
        gy = torch.arange(W, device=device, dtype=dtype) / W
        gx, gy = torch.meshgrid(gx, gy, indexing='ij')
        grid = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return grid

    def forward(self, x, timesteps=None, parameters=None):
        B, _, H, W  = x.shape
        grid        = self._grid(B, H, W, x.device, x.dtype)
        h           = self.lift(torch.cat([x, grid], dim=1))

        if not self.needs_cond:
            # M0 基线: 无调制分支,条件已在输入层拼接。
            for blk in self.blocks:
                h = blk(h)
        else:
            # FM pseudo-time embedding (accept both [B] and [B, 1]).
            if timesteps is None:
                t_emb = torch.zeros(B, self.time_dim, device=x.device, dtype=x.dtype)
            else:
                if timesteps.ndim == 2:
                    timesteps = timesteps.squeeze(-1)
                t_emb = self.t_mlp(_sinusoidal_time_embedding(timesteps, self.time_dim))

            # Physical-parameter embedding.
            if self.p_mlp is None:
                p_emb = None
            elif parameters is None:
                p_emb = torch.zeros(B, self.time_dim, device=x.device, dtype=x.dtype)
            else:
                p_emb = self.p_mlp(parameters)

            for blk in self.blocks:
                h = blk(h, t_emb=t_emb, p_emb=p_emb)

        out = self.proj(h)

        # Keep the FM vector field in the zero-mean psi tangent space.
        # (仅 psi 流函数空间需要；图像/通用状态请关闭该选项)
        if self.zero_mean_output:
            out = out - out.mean(dim=(-2, -1), keepdim=True)
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
# ConditionalFNO2D：条件向量场统一接口（theory/README.md §7.1 / §8）
# =============================================================================

class ConditionalFNO2D(nn.Module):
    """2D FNO 条件向量场 v_θ(x_s, s | c_t)，与 ConditionalUNet2D 接口完全一致。

    输入：
        x_s      [B, C, H, W]             FM 中间状态
        flow_time [B]                     FM 时间 s
        context  [B, K, C, H, W]          历史帧条件
    历史帧按 §8.1 合并为 K*C 通道后与 x_s 拼接（C + K*C + 可选 forcing/geometry）；
    输出与 x_s 完全同 shape（核心接口约束）。

    条件注入方式由 cond_mode 选择（13.txt §十三.A / 14.txt §四 的模型族）:
      - "global_film"（M1, 默认）: 标量条件 (s, dt, params) 经 sinusoidal
        embedding + channel-wise FiLM 注入每个 FNO block（s 作为 timesteps，
        dt 与 params 拼接后作为 parameters）;
      - "mode_film"（M2, 阶段 1）: 同上条件编码,但调制发生在频域、逐 mode:
        ŷ_o(k) = (1 + γ_o(k;c)) ⊙ (W_0(k) ⊛ x̂(k))_o [+ β_o(k;c)]，
        γ/β 由低秩 mode 基 φ_r(k) 生成,零初始化（初始退化为无条件 FNO）;
      - "cond_kernel"（M3, 阶段 2）: 直接条件化谱核本身:
        W(k;c) = W_0(k) + Σ_{r=1}^{R} a_r(c) W_r(k),a 零初始化 ->
        初始严格退化为固定核 FNO;核条件系数由 (s, dt, params) 生成
        （与 M1/M2 相同的条件编码）;
      - "slot"（阶段 3 系列,帧通道化以外的主模型）: 保留显式 slot 轴
        [B, T, C, H, W],只在 H, W 上做 2D FFT;底层为 FNO2dSlot,
        由 use_cond_kernel / use_mode_film / slot_rank 等 flag 选择
        M4（slot + 条件化谱核）/ M5（+ mode-wise FiLM）/ G 等变体;
      - "input_concat"（M0）: 标量条件 (s, dt, params) 广播到空间网格后
        直接与输入拼接: u'(x) = [x_s, u_{t-K+1:t}(x), s, dt, params](x)，
        FNO block 不携带任何调制分支（底层 net 以 film_kind="none" 构建）。

    Args:
        state_channels: 状态通道数 C
        context_len:    历史帧数 K
        param_dim:      物理参数维度 P（global/mode_film/cond_kernel 时 dt 自动并入，p_mlp 输入为 P+1）
        extra_channels: 额外条件通道数（forcing + geometry_mask，§8.1 可选）
        width / modes / depth: FNO 配置（README §8.2 推荐 width=64, modes=20, depth=4）
        zero_mean_output: 强制输出零空间均值（psi 流函数空间专用；图像状态默认 False）
        block_len:      联合生成块长度 L（默认 1：单帧状态，等价原行为）。
                        状态 = 整块未来帧 [B, C*L, H, W]，输入通道 = C*(L+K)，
                        输出通道 = C*L。
        cond_mode:      "global_film"（M1）| "mode_film"（M2）|
                        "cond_kernel"（M3）| "slot"（阶段 3 系列）|
                        "input_concat"（M0）
        film_rank:      mode-wise FiLM 的 mode 基数量 R（mode_film / slot 模式）
        use_beta:       是否启用频域加性偏置 β（默认关闭,13.txt §7.2）
        kernel_rank:    条件化谱核基数量 R（cond_kernel / slot 模式）;
        kernel_form:    "basis"（逐基核加权,默认）| "channel_low_rank"（U V^H）
        channel_low_rank: channel_low_rank 形式的秩 q（默认 in_channels//4）
        use_cond_kernel / use_mode_film / slot_rank / mode_basis /
        block_triangular / t0_init: 阶段 3 slot 模式 flag（见 FNO2dSlot）
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
            # 阶段 3 系列: 显式 slot 轴主模型（M4/M5/G/显式 slot 基线）
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
            # M2: 频域 mode-wise FiLM（阶段 1）
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
            # M3: 条件化谱核（阶段 2）
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
            # M1: 物理空间 channel-wise FiLM（现有行为）
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
            x = torch.zeros(B, self.block_len * C, H, W,
                            device=context.device, dtype=context.dtype)
        if self.cond_mode == "slot":
            # 阶段 3: 显式 slot 轴,整块直通（不做帧通道化展平）
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

    forward 签名与 ConditionalFNO2D 完全一致(x, flow_time, context, params,
    dt, ...),所有条件参数被忽略;FM 模式等价于忽略 s 的退化向量场
    (不建议 FM 使用,基线定位为 det 整块回归)。底层复用 FNO2dParam
    (film_kind="none" + kernel_kind="none" + param_dim=0 -> needs_cond=False,
    完全无条件的截断谱卷积 FNO)。
    """

    def __init__(self, state_channels: int = 2, context_len: int = 6,
                 block_len: int = 6, param_dim: int = 1,
                 image_size: int = 128, width: int = 64, modes: int = 20,
                 depth: int = 4, time_dim: int = 128,
                 zero_mean_output: bool = False):
        super().__init__()
        self.state_channels = state_channels
        self.context_len = context_len
        self.block_len = block_len
        self.param_dim = param_dim
        self.net = FNO2dParam(
            in_channels=state_channels * (context_len + block_len),
            out_channels=state_channels * block_len,
            width=width, depth=depth, modes=modes,
            time_dim=time_dim, param_dim=0,
            film_kind="none", kernel_kind="none",
            zero_mean_output=zero_mean_output,
        )

    def forward(self, x, flow_time, context, params=None, dt=None,
                forcing=None, geometry_mask=None):
        B, K, C, H, W = context.shape
        if x is None:
            x = torch.zeros(B, self.block_len * C, H, W,
                            device=context.device, dtype=context.dtype)
        context_flat = context.reshape(B, K * C, H, W)
        extra = [e for e in (forcing, geometry_mask) if e is not None]
        x_input = torch.cat([x, context_flat] + extra, dim=1)
        return self.net(x_input)


# =============================================================================
# 阶段 3: Slot-mode 条件调制（14.txt M4/M5 / 13.txt §7.5）
# =============================================================================

class SlotMix2d(nn.Module):
    """Slot-mode 双条件调制（阶段 3）:

        T_{l,m}(k;c) = T0_{l,m} + Σ_{r=1}^{R_s} a_r(c) A^(r)_{l,m} φ_r(k)

    - T0: 基础 slot 混合 [S, S]（S = K + L）。identity 初始化:
      T_CC = I, T_XX = I, T_XC = 1/K（均匀）, T_CX = 0;
    - A_r: slot 基 [R, S, S];φ_r: mode 基 [R, M1+1, M2]（行索引 = kx,
      与 ModeWiseFilm2d 相同的 Hermitian 配对翻转构造）;
    - a_r(c) 由条件投影生成,零初始化 -> T = T0（初始仅基础混合）;
    - block_triangular: T[out∈C, in∈X] = 0（14.txt §四,未来不反哺 context）;
    - mode_basis=False: 退化为 mode 无关的 T_{l,m}(c)（消融对照）。
    """

    def __init__(self, n_context: int, n_future: int, modes1: int, modes2: int,
                 cond_dim: int, slot_rank: int = 4, mode_basis: bool = True,
                 block_triangular: bool = True, t0_init: str = "identity"):
        super().__init__()
        K, L, S = n_context, n_future, n_context + n_future
        self.n_context = n_context
        self.n_future  = n_future
        self.modes1    = modes1
        self.modes2    = modes2
        self.cond_dim  = cond_dim
        self.slot_rank = slot_rank
        self.mode_basis = mode_basis

        mask = torch.ones(S, S)
        if block_triangular:
            mask[:K, K:] = 0.0                     # T_CX = 0
        self.register_buffer("mask", mask)

        if t0_init == "identity":
            T0 = torch.zeros(S, S)
            T0[:K, :K] = torch.eye(K)
            T0[K:, K:] = torch.eye(L)
            T0[K:, :K] = 1.0 / max(K, 1)
        elif t0_init == "random":
            T0 = 0.1 * torch.randn(S, S)
        else:
            raise ValueError(f"未知 t0_init: {t0_init}")
        self.T0 = nn.Parameter(T0 * mask)

        if slot_rank > 0:
            self.A = nn.Parameter(0.1 * torch.randn(slot_rank, S, S) * mask)
            if mode_basis:
                # 已经证明mode_basis有损结果，后续不会使用mode_basis，故直接删除初始化尺度校准
                self.phi = nn.Parameter(0.1 * torch.randn(slot_rank, modes1 + 1, modes2))
            else:
                self.phi = None
            self.a_proj = nn.Linear(cond_dim, slot_rank)
            nn.init.zeros_(self.a_proj.weight)
            nn.init.zeros_(self.a_proj.bias)

    @staticmethod
    def _halves(grid):
        """kx ∈ [0, M1] 网格 -> 上下半块（Hermitian 配对,同 ModeWiseFilm2d）。"""
        M1 = grid.shape[-2] - 1
        return grid[..., :M1, :], torch.flip(grid, dims=[-2])[..., :M1, :]

    def raw_mask_violation(self):
        """原始参数的禁止块(|T_CX|)最大绝对值——优化器污染检测。

        返回 (T0 违规, A 违规);forward 中强制 mask,故原始参数即使被
        污染也不影响有效算子(见 effective_mask_violation)。
        """
        forbidden = (1.0 - self.mask)
        t0_v = float((self.T0.abs() * forbidden).max().item())
        a_v = float((self.A.abs() * forbidden).max().item()) \
            if self.slot_rank > 0 else 0.0
        return t0_v, a_v

    def effective_mask_violation(self):
        """forward 中实际参与算子的禁止块违规——按有效矩阵真计算
        (6.txt §三.1: 不硬编码,避免 forward 重构后诊断失真)。"""
        forbidden = (1.0 - self.mask)
        t0_v = float(((self.T0 * self.mask).abs() * forbidden).max().item())
        if self.slot_rank > 0:
            a_v = float(((self.A * self.mask).abs() * forbidden).max().item())
        else:
            a_v = 0.0
        return t0_v, a_v

    # 兼容旧名(等价于 raw_mask_violation)
    def mask_violation(self):
        return self.raw_mask_violation()

    def forward(self, z_top, z_bot, cond):
        """z_top / z_bot: [B, S, C, M1', M2'] 截断谱块 -> 混合后谱块。

        沿 slot 维混合: q[b,o,c,x,y] = Σ_m T[b,o,m,x,y] z[b,m,c,x,y]。
        块三角约束在 forward 中强制(4.txt §四.3): T0/A 参数可能被优化器
        污染,因此每步都用 self.mask 屏蔽禁止块。
        """
        B, S, C, M1, M2 = z_top.shape
        M1 = min(self.modes1, M1)
        M2 = min(self.modes2, M2)
        T0 = self.T0 * self.mask
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
    """Slot 谱层（阶段 3,对应 14.txt §五 的谱层顺序）:

        空间 FFT -> 条件化谱核（共享,可选）-> slot-mode mixing T_{l,m}(k;c)
        -> mode-wise FiLM（逐 slot,可选）-> IFFT -> pointwise -> GELU

    h_C [B, K, D, H, W] 与 h_X [B, L, D, H, W] 拼接为 S = K + L 个 slot 一起
    处理;块三角 slot mixing 保证 future 组不反哺 context 组。
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
        self.channels   = channels
        self.modes1     = modes1
        self.modes2     = modes2
        self.cond_dim   = cond_dim
        self.n_context  = n_context
        self.n_future   = n_future
        self.use_cond_kernel = use_cond_kernel
        self.use_mode_film   = use_mode_film

        if use_cond_kernel:
            # 阶段 2: 条件化谱核（跨 slot 共享,slot 依赖由 mixing 表达）
            self.spectral = ConditionalSpectralConv2d(
                channels, channels, modes1, modes2, cond_dim,
                basis_rank=kernel_rank, kernel_form=kernel_form,
                channel_low_rank=channel_low_rank,
                kernel_coeff_max=kernel_coeff_max)
        else:
            self.spectral = SpectralConv2d(channels, channels, modes1, modes2)
        # 阶段 3: slot-mode mixing
        self.slot_mix = SlotMix2d(
            n_context, n_future, modes1, modes2, cond_dim,
            slot_rank=slot_rank, mode_basis=mode_basis,
            block_triangular=block_triangular, t0_init=t0_init)
        # 阶段 1: mode-wise FiLM（逐 slot 系数,use_mode_film=False 时为恒等）
        self.mode_film = ModeWiseFilm2d(
            channels, modes1, modes2, cond_dim,
            rank=film_rank if use_mode_film else 0, use_beta=use_beta,
            per_slot=n_context + n_future, gamma_max=gamma_max)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, h_C, h_X, cond):
        B, K, D, H, W = h_C.shape
        h   = torch.cat([h_C, h_X], dim=1)                  # [B, S, D, H, W]
        S   = h.shape[1]
        W2  = W // 2 + 1
        h_ft = torch.fft.rfft2(h, norm="ortho")             # [B, S, D, H, W2]
        if self.use_cond_kernel:
            cond_rep = cond.repeat_interleave(S, dim=0)     # 共享核: 各 slot 同系数
            z = self.spectral.apply_ft(h_ft.reshape(B * S, D, H, W2), cond_rep)
        else:
            z = self.spectral.apply_ft(h_ft.reshape(B * S, D, H, W2))
        z = z.reshape(B, S, D, H, W2)
        m1  = min(self.modes1, H // 2)
        m2  = min(self.modes2, W2)
        q_top, q_bot = self.slot_mix(z[:, :, :, :m1, :m2], z[:, :, :, -m1:, :m2], cond)
        q_top, q_bot = self.mode_film(q_top, q_bot, cond)
        q_ft = torch.zeros_like(z)
        q_ft[:, :, :, :m1, :m2]  = q_top
        q_ft[:, :, :, -m1:, :m2] = q_bot
        q = torch.fft.irfft2(q_ft, s=(H, W), norm="ortho")
        p = self.pointwise(h.reshape(B * S, D, H, W)).reshape(B, S, D, H, W)
        h_out = self.act(q + p)
        return h_out[:, :K], h_out[:, K:]


class FNO2dSlot(nn.Module):
    """显式 slot 轴条件谱算子（阶段 3 主模型,14.txt §三/§四/§五）。

    - 只在 H, W 上做 2D FFT;时间维保留为显式 slot 轴（非 3D FNO）;
    - context 组 [B, K, C, H, W] 与 future 组 [B, L, C, H, W] 经**共享 lift**
      (5.txt §八.1: 同一 Conv2d,逐 slot 共享升维)投影到 width,
      块三角 slot mixing T = [[T_CC, 0], [T_XC, T_XX]];
    - 每层: 空间 FFT -> 条件化谱核（共享,可选）-> T_{l,m}(k;c) slot mixing
      -> mode-wise FiLM（逐 slot,可选）-> IFFT -> pointwise;
    - FM 接口与 ConditionalFNO2D 完全一致:
        forward(x [B, C*L, H, W], flow_time [B], context, params, dt) -> 同 shape
      x 即 FM 中间状态整块,reshape 为 [B, L, C, H, W] 作为 future 组输入;
      x=None 时 future 组置零（确定性整块回归路径）。

    Args:
        state_channels: 状态通道数 C;context_len: 历史帧数 K;block_len: 块长 L
        param_dim:      物理参数维度 P（dt 自动并入,p_mlp 输入为 P+1）
        use_cond_kernel: 条件化谱核（阶段 2,默认关闭）
        use_mode_film:   mode-wise FiLM（阶段 1,默认关闭）
        slot_rank:       slot mixing 条件秩 R_s（0 = 仅 T0,无条件混合）
        mode_basis:      slot mixing 是否 mode 依赖（False = T_{l,m}(c)）
        block_triangular: 块三角结构（默认 True）
        t0_init:         T0 初始化 "identity" | "random"
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
        self.state_channels = state_channels
        self.context_len = context_len
        self.block_len = block_len
        self.param_dim = param_dim
        self.extra_channels = extra_channels
        self.width = width
        self.time_dim = time_dim
        self.zero_mean_output = zero_mean_output
        self.use_cond_kernel = use_cond_kernel
        self.use_mode_film = use_mode_film

        # 条件编码（与 ConditionalFNO2D 同约定: s -> t_emb, [dt, params] -> p_emb）
        self.t_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.p_mlp = nn.Sequential(
            nn.Linear(param_dim + 1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))

        # 共享 lift（逐 slot 施加,5.txt §八.1）: 状态 C + 额外条件 + grid(2)
        self.lift = nn.Conv2d(state_channels + extra_channels + 2, width, kernel_size=1)
        cond_dim = 2 * time_dim                      # [t_emb, p_emb] 拼接
        self.blocks = nn.ModuleList([
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
        # 零初始化最终投影（FM 从零场出发,与 FNO2dParam 同约定）
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    @staticmethod
    def _grid(B: int, H: int, W: int, device, dtype):
        gx = torch.arange(H, device=device, dtype=dtype) / H
        gy = torch.arange(W, device=device, dtype=dtype) / W
        gx, gy = torch.meshgrid(gx, gy, indexing='ij')
        grid = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return grid

    def forward(self, x, flow_time, context, params=None, dt=None,
                forcing=None, geometry_mask=None):
        B, K, C, H, W = context.shape
        L = self.block_len
        if x is None:
            x_slots = torch.zeros(B, L, C, H, W, device=context.device, dtype=context.dtype)
        else:
            x_slots = x.reshape(B, L, C, H, W)          # FM 中间状态整块 -> 未来 slot

        grid   = self._grid(B, H, W, context.device, context.dtype)
        extras = [e for e in (forcing, geometry_mask) if e is not None]
        ctx_in = torch.cat(
            [context.reshape(B * K, C, H, W), grid.repeat_interleave(K, 0)]
            + [e.repeat_interleave(K, 0) for e in extras], dim=1)
        fut_in = torch.cat(
            [x_slots.reshape(B * L, C, H, W), grid.repeat_interleave(L, 0)]
            + [e.repeat_interleave(L, 0) for e in extras], dim=1)
        # 输入一致性断言(5.txt §八.3: extra_channels>0 时必须传入 forcing/geometry_mask)
        assert ctx_in.shape[1] == self.lift.in_channels, \
            f"ctx_in 通道 {ctx_in.shape[1]} != lift.in_channels {self.lift.in_channels}"
        assert fut_in.shape[1] == self.lift.in_channels, \
            f"fut_in 通道 {fut_in.shape[1]} != lift.in_channels {self.lift.in_channels}"
        h_C = self.lift(ctx_in).reshape(B, K, self.width, H, W)
        h_X = self.lift(fut_in).reshape(B, L, self.width, H, W)

        # 条件: s -> t_emb;[dt, params] -> p_emb(缺省补零,避免 shape 错配)
        if flow_time is None:
            t_emb = torch.zeros(B, self.time_dim, device=context.device, dtype=context.dtype)
        else:
            if flow_time.ndim == 2:
                flow_time = flow_time.squeeze(-1)
            t_emb = self.t_mlp(_sinusoidal_time_embedding(flow_time, self.time_dim))
        dt_v  = dt.reshape(B, 1) if dt is not None else \
            torch.zeros(B, 1, device=context.device, dtype=context.dtype)
        p_v   = params.reshape(B, -1) if params is not None else \
            torch.zeros(B, self.param_dim, device=context.device, dtype=context.dtype)
        p_emb = self.p_mlp(torch.cat([dt_v, p_v], dim=-1))
        cond  = torch.cat([t_emb, p_emb], dim=-1)       # [B, 2*time_dim]

        for blk in self.blocks:
            h_C, h_X = blk(h_C, h_X, cond)

        out = self.proj(h_X.reshape(B * L, self.width, H, W)).reshape(B, L, C, H, W)
        out = out.reshape(B, C * L, H, W)
        if self.zero_mean_output:
            out = out - out.mean(dim=(-2, -1), keepdim=True)
        return out


# =============================================================================
# 模型族 M0 - M5（13.txt §十三.A / 14.txt §四）:
#   在 ConditionalFNO2D（FM 联合生成接口,含流时间 s）上选择条件注入方式。
# =============================================================================

class FNO2dInputConcat(ConditionalFNO2D):
    """M0 基线: FNO-InputConcat —— 条件 (s, dt, params) 广播拼接,无调制。

    u'(x) = [x_s, u_{t-K+1:t}(x), s, dt, params](x)
    底层 net 以 film_kind="none" 构建: FNO block 为纯无条件谱层。
    """

    def __init__(self, **kwargs):
        kwargs["cond_mode"] = "input_concat"
        super().__init__(**kwargs)


class FNO2dGlobalFilm(ConditionalFNO2D):
    """M1 基线: FNO-GlobalFiLM —— 物理空间 channel-wise FiLM。

    (s, dt, params) 经 embedding + FiLM 注入每个 FNO block。
    """

    def __init__(self, **kwargs):
        kwargs["cond_mode"] = "global_film"
        super().__init__(**kwargs)


class FNO2dModeFilm(ConditionalFNO2D):
    """M2（阶段 1）: FNO-ModeFiLM —— 频域 mode-wise FiLM。

        ŷ_o(k) = (1 + γ_o(k;c)) ⊙ (W_0(k) ⊛ x̂(k))_o [+ β_o(k;c)]

    γ/β 由低秩 mode 基生成（film_rank 个 φ_r(k)），零初始化 -> 初始
    等价于无条件 FNO。条件编码与 M1 相同（s 作 timesteps, [dt, params]
    作 parameters）。
    """

    def __init__(self, **kwargs):
        kwargs["cond_mode"] = "mode_film"
        super().__init__(**kwargs)


class FNO2dCondKernel(ConditionalFNO2D):
    """M3（阶段 2）: FNO-ConditionalKernel —— 条件化谱核。

        ŷ(k) = (W_0(k) + Σ_{r=1}^{R} a_r(c) W_r(k)) ⊛ x̂(k)

    条件系数 a_r 零初始化 -> 初始严格退化为固定核 FNO。条件编码与 M1 相同
    （s 作 timesteps, [dt, params] 作 parameters）。可选用
    kernel_form="channel_low_rank"（W_r = U_r V_r^H）降低参数/显存。
    """

    def __init__(self, **kwargs):
        kwargs["cond_mode"] = "cond_kernel"
        super().__init__(**kwargs)


class FNO2dModeFilmCondKernel(ConditionalFNO2D):
    """消融 D 有界版(18.txt 实验 1): ModeFiLM-Bnd + 条件化谱核。

        ŷ(k) = (1 + γ_o(k;c)) ⊙ (W_0(k) + Σ_r a_r(c) W_r(k)) ⊛ x̂(k)

    同一 block 内先条件化谱核再 mode-wise 有界调制;film 与核系数共用
    同一 cond 编码([s, dt, params])。kernel_form 默认 channel_low_rank,
    γ 由 gamma_max 有界。
    """

    def __init__(self, **kwargs):
        kwargs["cond_mode"] = "mode_film_kernel"
        super().__init__(**kwargs)


class FNO2dSlotCondKernel(ConditionalFNO2D):
    """M4（阶段 3）: Slot-ConditionalKernel —— 显式 slot + 条件化谱核。

        ŷ_l(k) = Σ_m T_{l,m}(k;c) W_eff(k;c) x̂_m(k)
    显式 slot 轴（2D FFT）,块三角结构,条件化谱核 + slot-mode mixing;
    无 mode-wise FiLM 校准。
    """

    def __init__(self, **kwargs):
        kwargs["cond_mode"] = "slot"
        kwargs.setdefault("use_cond_kernel", True)
        kwargs.setdefault("use_mode_film", False)
        super().__init__(**kwargs)


class FNO2dSlotFull(ConditionalFNO2D):
    """M5（完整模型）: Mode FiLM + 条件化谱核 + Slot-mode 全叠。

        ŷ_l(k) = D_{γ,l}(k;c) Σ_m T_{l,m}(k;c) [W_0(k) + ΔW(k;c)] x̂_m(k)
    """

    def __init__(self, **kwargs):
        kwargs["cond_mode"] = "slot"
        kwargs.setdefault("use_cond_kernel", True)
        kwargs.setdefault("use_mode_film", True)
        super().__init__(**kwargs)


class FNO2dSlotModeFilm(ConditionalFNO2D):
    """G 变体: 显式 slot + mode-wise FiLM（无条件化谱核）。

    slot mixing + 逐 slot mode film,固定谱核——消融组合 G
    （Mode-FiLM 与 Slot-mode 的组合能力）。
    """

    def __init__(self, **kwargs):
        kwargs["cond_mode"] = "slot"
        kwargs.setdefault("use_cond_kernel", False)
        kwargs.setdefault("use_mode_film", True)
        super().__init__(**kwargs)


# 尺寸模板（与 FNO_PARAM_CONFIGS 的 s/b/l 同构）
COND_FNO_CONFIGS = {
    "s": dict(width=32,  modes=12, depth=4, time_dim=128),
    "b": dict(width=64,  modes=20, depth=4, time_dim=128),
    "l": dict(width=96,  modes=24, depth=6, time_dim=128),
}


def build_cond_fno(cond_mode: str = "global_film", size: str = "b",
                   state_channels: int = 1, context_len: int = 4,
                   block_len: int = 1, param_dim: int = 1,
                   image_size: int = 64, **kwargs):
    """构造条件 FNO 模型族 M0-M5（ConditionalFNO2D 接口,含 flow time s）。

    Args:
        cond_mode: "input_concat"（M0）| "global_film"（M1）| "mode_film"（M2）
                   | "cond_kernel"（M3）| "slot"（显式 slot 基线）|
                   "slot_kernel"（M4）| "slot_full"（M5）| "slot_film"（G）
        size:      s / b / l 模板（COND_FNO_CONFIGS）
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
           "cond_kernel":  FNO2dCondKernel,
           "mode_film_kernel": FNO2dModeFilmCondKernel,
           "slot":         FNO2dSlot,
           "slot_kernel":  FNO2dSlotCondKernel,
           "slot_full":    FNO2dSlotFull,
           "slot_film":    FNO2dSlotModeFilm}[cond_mode]
    return cls(state_channels=state_channels, context_len=context_len,
               block_len=block_len, param_dim=param_dim,
               image_size=image_size, **cfg)


# =============================================================================
# 自检（不依赖训练脚本）: python models/cfno.py
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    B, K, L, C, P, H, W = 2, 3, 4, 2, 1, 64, 64
    for mode in ("input_concat", "global_film", "mode_film", "cond_kernel",
                 "slot", "slot_kernel", "slot_full", "slot_film"):
        model = build_cond_fno(mode, size="s", state_channels=C, context_len=K,
                               block_len=L, param_dim=P)
        x_s = torch.randn(B, C * L, H, W)
        ctx = torch.randn(B, K, C, H, W)
        s   = torch.rand(B)
        dt  = torch.rand(B, 1)
        p0  = torch.zeros(B, P)
        p1  = torch.randn(B, P)
        out = model(x_s, s, ctx, params=p0, dt=dt)
        assert out.shape == x_s.shape, f"输出形状 {tuple(out.shape)} 与 x_s 不一致"
        if mode not in ("input_concat",):
            # 零初始化 -> 初始向量场与 s/dt/params 无关(退化性质)。
            # 注意: 因最终投影零初始化,该检查对"条件分支正确性"是必要非充分
            # (5.txt §四.1),真正的条件学习由下方"两步优化"测试覆盖。
            assert torch.allclose(
                out, model(x_s, s * 0.5, ctx, params=p1, dt=dt * 2), atol=1e-6), \
                f"{mode} 初始输出不应依赖 s/dt/params"
        loss = out.square().mean()
        loss.backward()
        grads = [p.grad is not None for p in model.parameters()
                 if p.requires_grad]
        assert all(grads), "存在无梯度的参数"
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{mode}] shape={tuple(out.shape)} params={n_params:,} OK")

    # ---- 阶段 3 专项: 块三角结构 + x=None 确定性路径 ----
    model = FNO2dSlot(state_channels=C, context_len=K, block_len=L, param_dim=P,
                      width=16, modes=8, depth=2)
    tm = model.blocks[0].slot_mix
    T0 = tm.T0.detach()
    S  = K + L
    assert torch.allclose(T0[:K, K:], torch.zeros(K, L)), "T_CX 应为 0(块三角)"
    assert torch.allclose(T0[:K, :K], torch.eye(K)), "T_CC 应为 I"
    assert torch.allclose(T0[K:, K:], torch.eye(L)), "T_XX 应为 I"
    assert torch.allclose(T0[K:, :K], torch.full((L, K), 1.0 / K)), "T_XC 应为 1/K"
    out_det = model(None, None, torch.randn(B, K, C, H, W), params=p0, dt=dt)
    assert out_det.shape == (B, C * L, H, W)
    print("slot: block-triangular T0 OK | x=None deterministic path OK")

    # ---- 小尺寸单元测试(6.txt §五: 数学测试缩小到 16²/w8,CI 友好)----
    b, hh, ww, wd = 2, 16, 16, 8

    # 1) channel_low_rank 数学等价(随机非零 a,5.txt §六.1)
    cs = ConditionalSpectralConv2d(wd, wd, 4, 3, cond_dim=4, basis_rank=2,
                                   kernel_form="channel_low_rank",
                                   channel_low_rank=2)
    xx = torch.randn(b, wd, 4, 3, dtype=torch.cfloat)
    aa = torch.randn(b, 2)
    dtop, dbot = cs._kernel_apply(xx, xx, aa)
    for tag, U, V, d in [("top", cs.u1, cs.v1, dtop), ("bot", cs.u2, cs.v2, dbot)]:
        dW = torch.zeros(b, wd, wd, 4, 3, dtype=torch.cfloat)
        for r in range(2):
            dW = dW + aa[:, r].view(b, 1, 1, 1, 1) * torch.einsum(
                "iqxy,oqxy->ioxy", U[r], V[r].conj())
        d_exp = torch.einsum("bixy,bioxy->boxy", xx, dW)
        assert torch.allclose(d, d_exp, atol=1e-5), f"lowrank {tag} 不等价"
    print("self-check: channel_low_rank 公式等价 OK")

    # 2) mode_basis=False 与 slot_rank=0 前向/反向(5.txt §二)
    xs16 = torch.randn(b, C * L, hh, ww)
    ctx16 = torch.randn(b, K, C, hh, ww)
    for mb, sr in [(False, 2), (True, 0), (False, 0)]:
        m2 = FNO2dSlot(state_channels=C, context_len=K, block_len=L, param_dim=P,
                       width=wd, modes=4, depth=2, mode_basis=mb, slot_rank=sr)
        o = m2(xs16, s, ctx16, params=p1, dt=dt)
        o.square().mean().backward()
        assert o.shape == xs16.shape
        if sr > 0:
            assert m2.blocks[0].slot_mix.raw_mask_violation() == (0.0, 0.0)
            assert m2.blocks[0].slot_mix.effective_mask_violation() == (0.0, 0.0)
    print("self-check: mode_basis=False / slot_rank=0 fwd+bwd OK")

    # 3) 块三角 Jacobian: ∂h_C'/∂h_X = 0,训练步前后各测一次(6.txt §四.2)
    blk = FNO2dSlot(state_channels=C, context_len=K, block_len=L, param_dim=P,
                    width=wd, modes=4, depth=2).blocks[0]
    hC = torch.randn(b, K, wd, hh, ww, requires_grad=True)
    hX = torch.randn(b, L, wd, hh, ww, requires_grad=True)
    cond = torch.randn(b, blk.cond_dim)

    def jacobian_check():
        outC, _ = blk(hC, hX, cond)
        gX = torch.autograd.grad(outC.sum(), hX, allow_unused=True)[0]
        assert gX is None or float(gX.abs().max()) == 0.0, "∂h_C/∂h_X 应为 0"

    jacobian_check()
    opt_j = torch.optim.Adam(blk.parameters(), lr=1e-3)
    for _ in range(2):
        opt_j.zero_grad()
        outC, _ = blk(hC, hX, cond)
        outC.square().mean().backward()
        opt_j.step()
    jacobian_check()
    print("self-check: 块三角 Jacobian ∂h_C/∂h_X=0(训练步前后) OK")

    # 4) 条件分支更新 + 训练后输出条件敏感性(6.txt §四.1:
    #    梯度存在 ≠ 输出对条件可测;用非零 target 训练 8 步后比较输出)
    def train_then_sens(mode, extra):
        mm = build_cond_fno(mode, size="s", state_channels=C, context_len=K,
                            block_len=L, param_dim=P, image_size=hh,
                            width=wd, modes=4, kernel_form="channel_low_rank",
                            kernel_rank=2, film_rank=2, **extra)
        target = torch.randn(b, C * L, hh, ww)
        opt = torch.optim.Adam(mm.parameters(), lr=1e-3)
        for _ in range(16):
            opt.zero_grad()
            loss = (mm(xs16, s, ctx16, params=p1, dt=dt) - target).square().mean()
            loss.backward()
            opt.step()
        # 条件分支梯度: 存在有限非零值(放宽为 阈值+计数,6.txt §五)
        grads = [p.grad for p in mm.parameters() if p.requires_grad and p.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads), "存在非有限梯度"
        assert sum(float(g.norm()) > 1e-12 for g in grads) >= 10, \
            f"{mode} 条件分支梯度不足"
        with torch.no_grad():
            y0 = mm(xs16, s, ctx16, params=p0, dt=dt)
            y1 = mm(xs16, s * 0.5, ctx16, params=p1, dt=dt * 2)
            delta = float((y0 - y1).abs().mean())
        assert delta > 1e-8, f"{mode} 训练后输出对条件不敏感 (Δ={delta:.2e})"
        return delta

    for mode, extra in [("mode_film", {}), ("cond_kernel", {}),
                        ("slot", {}), ("slot_kernel", {})]:
        d = train_then_sens(mode, extra)
        print(f"self-check: {mode} 条件敏感性 Δ={d:.2e} OK")

    # 5) 小分辨率截断路径(H=16 < 2*modes,5.txt §五.6)
    for mode in ("cond_kernel", "slot_kernel"):
        m4 = build_cond_fno(mode, size="s", state_channels=C, context_len=K,
                            block_len=L, param_dim=P, image_size=16,
                            kernel_form="channel_low_rank", kernel_rank=2)
        o = m4(torch.randn(B, C * L, 16, 16), s, torch.randn(B, K, C, 16, 16),
               params=p1, dt=dt)
        assert o.shape == (B, C * L, 16, 16)
    print("self-check: 小分辨率截断路径 OK")

    # 6) gamma_max 有界增益(5.txt §七)
    m5 = build_cond_fno("mode_film", size="s", state_channels=C,
                        context_len=K, block_len=L, param_dim=P,
                        film_rank=4, gamma_max=0.5)
    gm = m5.net.blocks[0].mode_film
    a = torch.randn(8, 1, gm.channels, gm.rank)   # 随机系数(绕过零初始化 a_proj)
    grid = torch.einsum("bgcr,rxy->bgcxy", a, gm.phi)
    assert float(grid.abs().max()) > 0.5, "测试增益应超出有界范围"
    bounded = gm.gamma_max * torch.tanh(grid)
    assert float(bounded.abs().max()) <= 0.5 + 1e-6, "有界增益应 ≤ γ_max"
    print("self-check: gamma_max 有界增益 OK")

    # 7) 初始化尺度不变量(当前行为: UV 校准与 φ 校准均已移除,
    #    7.txt 校准为一次性对照;尺度差异保留,见 14.txt 结论)
    #    a) lowrank 核: RMS(U_r V_r^H) 应显著小于 RMS(W_r^basis)
    #      (未校准: √q·(bscale√2)² << bscale√2,7.txt §八.1)
    csc = ConditionalSpectralConv2d(8, 8, 6, 5, cond_dim=4, basis_rank=2,
                                    kernel_form="channel_low_rank",
                                    channel_low_rank=3)
    csb = ConditionalSpectralConv2d(8, 8, 6, 5, cond_dim=4, basis_rank=2,
                                    kernel_form="basis")
    target_rms = float(csb.basis1[0].abs().square().mean().sqrt())
    for r in range(2):
        W_r = torch.einsum("iqxy,oqxy->ioxy", csc.u1[r], csc.v1[r].conj())
        rms_r = float(W_r.abs().square().mean().sqrt())
        assert rms_r / target_rms < 0.5, \
            f"lowrank 未校准尺度应小于 basis(ratio={rms_r / target_rms:.2f})"
    #    b) slot mixing(mode_basis=True): φ 保持 0.1·randn(不归一)
    smc = SlotMix2d(n_context=K, n_future=L, modes1=6, modes2=5,
                    cond_dim=8, slot_rank=2, mode_basis=True)
    for r in range(2):
        rms_phi = float(smc.phi[r].square().mean().sqrt())
        assert 0.05 < rms_phi < 0.2, f"φ_{r} RMS {rms_phi:.4f} 应≈0.1(不校准)"
    print("self-check: 初始化尺度不变量(不校准行为) OK")

    # 7b) 消融 D 有界版(18.txt 实验 1: mode_film_kernel)前向/反向
    mfk = build_cond_fno("mode_film_kernel", size="s", state_channels=C,
                         context_len=K, block_len=L, param_dim=P,
                         width=8, modes=4, film_rank=2, gamma_max=0.5,
                         kernel_rank=2, kernel_form="channel_low_rank",
                         channel_low_rank=2)
    o = mfk(torch.randn(b, C * L, 16, 16), s, torch.randn(b, K, C, 16, 16),
            params=p1, dt=dt)
    o.square().mean().backward()
    assert o.shape == (b, C * L, 16, 16)
    assert all(hasattr(blk, "mode_film") for blk in mfk.net.blocks)
    assert all(blk.kernel_rank > 0 for blk in mfk.net.blocks)
    print("self-check: mode_film_kernel(消融D有界) fwd+bwd OK")

    # 7c) 无条件 FNO 基线: 前向/反向 + 条件不敏感
    # (22.txt §2: 固定相同 x/context,仅改变被忽略的条件;并让输出层非零,
    #  避免零初始化投影导致的 vacuous 通过)
    uf = UncondFNO2d(state_channels=C, context_len=K, block_len=L, param_dim=P,
                     image_size=16, width=8, modes=4, depth=2)
    with torch.no_grad():
        uf.net.proj[-1].weight.normal_(mean=0.0, std=1e-2)
        uf.net.proj[-1].bias.normal_(mean=0.0, std=1e-2)
    x_uf = torch.randn(b, C * L, 16, 16)
    ctx_uf = torch.randn(b, K, C, 16, 16)
    o1 = uf(x_uf, s, ctx_uf, params=p0, dt=dt)
    o1.square().mean().backward()
    assert o1.shape == (b, C * L, 16, 16)
    assert float(o1.detach().abs().mean()) > 1e-6, "测试输出不应为零(vacuous)"
    with torch.no_grad():
        o2 = uf(x_uf, s * 0.3, ctx_uf, params=p1, dt=dt * 2)
    assert torch.allclose(o1.detach(), o2, atol=1e-6), \
        "UncondFNO2d 不应依赖 flow_time / params / dt"
    print("self-check: UncondFNO2d fwd+bwd/条件不敏感(固定条件输入) OK")

    # 7d) kernel_coeff_max: 条件核系数门控(22.txt §3)
    kmax = 0.5
    ck = ConditionalSpectralConv2d(wd, wd, 4, 3, cond_dim=4, basis_rank=2,
                                   kernel_form="channel_low_rank",
                                   channel_low_rank=2, kernel_coeff_max=kmax)
    with torch.no_grad():
        ck.a_proj.weight.fill_(10.0)
        ck.a_proj.bias.fill_(10.0)
    cond_test = torch.ones(b, 4)
    a_raw = ck.a_proj(cond_test)
    a_bnd = ck.kernel_coeff_max * torch.tanh(a_raw)
    assert float(a_raw.abs().max()) > kmax, "原始系数应超出 bound"
    assert float(a_bnd.abs().max()) <= kmax + 1e-6, "|a_r(c)| <= kernel_coeff_max"
    xft = torch.randn(b, wd, 8, 5, dtype=torch.cfloat)
    m1, m2 = 4, 3
    xt = xft[:, :, :m1, :m2]
    xb = xft[:, :, -m1:, :m2]
    out_t = ck._compl_mul2d(xt, ck.w0_1[:, :, :m1, :m2])
    out_b = ck._compl_mul2d(xb, ck.w0_2[:, :, :m1, :m2])
    dt7, db7 = ck._kernel_apply(xt, xb, a_bnd)
    expected = torch.zeros(b, wd, 8, 5, dtype=torch.cfloat)
    expected[:, :, :m1, :m2] = out_t + dt7
    expected[:, :, -m1:, :m2] = out_b + db7
    actual = ck.apply_ft(xft, cond_test)
    assert torch.allclose(actual, expected, atol=1e-5), \
        "apply_ft 未使用 bounded kernel coefficients"
    print("self-check: kernel_coeff_max 系数门控 OK")

    # 7e) kernel_coeff_max 高层透传(22.txt §4)
    kbase = dict(size="s", state_channels=C, context_len=K, block_len=L,
                 param_dim=P, width=8, modes=4, kernel_rank=2,
                 kernel_form="channel_low_rank", channel_low_rank=2,
                 kernel_coeff_max=kmax)
    m_ck = build_cond_fno("cond_kernel", **kbase)
    assert all(blk.spectral.kernel_coeff_max == kmax for blk in m_ck.net.blocks), \
        "cond_kernel 的 kernel_coeff_max 未透传"
    m_sk = build_cond_fno("slot_kernel", **kbase)
    assert all(blk.spectral.kernel_coeff_max == kmax for blk in m_sk.net.blocks), \
        "slot_kernel 的 kernel_coeff_max 未透传"
    m_mfk = build_cond_fno("mode_film_kernel", film_rank=2, gamma_max=0.5, **kbase)
    assert all(blk.spectral.kernel_coeff_max == kmax for blk in m_mfk.net.blocks), \
        "mode_film_kernel 的 kernel_coeff_max 未透传"
    print("self-check: kernel_coeff_max 高层透传 OK")

    print("self-check OK")

