
import os
import numpy as np
import torch
import torch.distributed as dist
import logging
from collections import OrderedDict
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchdiffeq import odeint
import matplotlib.pyplot as plt


# =============================================================================
#  Base-noise samplers for Flow Matching in stream-function (psi) space.
# =============================================================================

# ----------------------------- task-specific noise ------------------------- #
def make_noise_sampler(args):
    """x_0 基噪声采样器

    - 'white'      : 零均值空间白噪声
    """
    if args.noise_type == "white":
        def sampler(x):
            n = torch.randn_like(x)
            return n - n.mean(dim=(-2, -1), keepdim=True)
        return sampler
    else:
        raise ValueError(f"Unknown noise type: {args.noise_type}")


@torch.no_grad()
def _wavenumber_grid(H: int, W: int, domain_size: float, device):
    """Angular wavenumber |k| grid

    k = 2*pi * fftfreq(N, d=L/N), so the physical meaning is spatial angular frequency (rad / unit length).
    """
    dx = float(domain_size) / W
    dy = float(domain_size) / H
    kx = 2.0 * torch.pi * torch.fft.fftfreq(W, d=dx, device=device)
    ky = 2.0 * torch.pi * torch.fft.fftfreq(H, d=dy, device=device)
    ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")
    return torch.sqrt(kx_grid**2 + ky_grid**2)  # (H, W)

def compute_metrics(pred, target):
    """
    Compute MSE, MAE, PSNR, and SSIM between prediction and target.
    
    Args:
        pred: [B, C, H, W] tensor
        target: [B, C, H, W] tensor
        
    Returns:
        metrics: dict containing score values
    """
    # MSE and MAE
    mse         = torch.mean((pred - target) ** 2).item()
    mae         = torch.mean(torch.abs(pred - target)).item()
    
    # Data range for PSNR and SSIM
    data_range  = (target.max() - target.min()).item()
    
    # PSNR
    psnr_metric = PeakSignalNoiseRatio(data_range=data_range).to(pred.device)
    psnr        = psnr_metric(pred, target).item()
    
    # SSIM
    ssim_metric = StructuralSimilarityIndexMeasure(
        data_range=data_range,
        kernel_size=11,
        sigma=1.5,
        k1=0.01,
        k2=0.03,
        gaussian_kernel=True
    ).to(pred.device)
    ssim = ssim_metric(pred, target).item()
    
    return {
        'MSE': mse,
        'MAE': mae,
        'PSNR': psnr,
        'SSIM': ssim
    }


from dataset_loader import scale_values

def to_physical_velocity(x: torch.Tensor) -> torch.Tensor:
    scales = torch.tensor(scale_values, device=x.device, dtype=x.dtype).view(1, len(scale_values), 1, 1)
    return x[:, : len(scale_values)] * scales

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    ema_params   = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def torchdiffeq_method(name):
    return {"heun": "heun2"}.get(name, name) # 非 heun 外,返回 name 本身



@torch.no_grad()
def compute_dataset_std(dataset, channel_wise=False):
    """Compute the global standard deviation of HR velocity fields in a dataset.

    Args:
        dataset: torch Dataset where each item is ``(cond, x_hr)``.
        channel_wise: If True, return per-channel std as a 1-D tensor.

    Returns:
        Python float (or 1-D tensor when ``channel_wise=True``).
    """
    from torch.utils.data import Subset, TensorDataset
    base, idx = dataset, None
    if isinstance(base, Subset):
        idx = list(base.indices)
        base = base.dataset
    if isinstance(base, TensorDataset) and len(base.tensors) >= 2:
        x = base.tensors[1] if idx is None else base.tensors[1][idx]
        if channel_wise:
            return x.transpose(0, 1).reshape(x.shape[1], -1).std(dim=1)
        return float(x.std())
    # Streaming fallback (works for any Dataset).
    n, s, sq = 0, 0.0, 0.0
    for _, x in dataset:
        x = x.float()
        n += x.numel()
        s += float(x.sum())
        sq += float((x * x).sum())
    mean = s / max(n, 1)
    var = max(sq / max(n, 1) - mean * mean, 0.0)
    return float(var ** 0.5)


class EarlyStopper:
    """Classic early stopping helper.

    Args:
        patience:  number of validations to wait without improvement.
        min_delta: minimum absolute change considered as improvement.
        mode:      'min' (loss) or 'max' (e.g. PSNR).
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.0, mode: str = "min"):
        assert mode in ("min", "max")
        self.patience = patience
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.bad_epochs = 0
        self.improved = False

    def step(self, value: float) -> bool:
        """Returns True if training should stop."""
        if self.mode == "min":
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta
        self.improved = improved
        if improved:
            self.best = float(value)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def state_dict(self):
        return {"best": self.best, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state):
        self.best = state["best"]
        self.bad_epochs = state["bad_epochs"]


def cleanup_old_files(pattern: str, keep_last_n: int = 4, logger=None,
                      epoch_extractor=None):
    """Keep only the latest ``keep_last_n`` files matching ``pattern``.
    """
    import glob as _glob, os as _os
    files = _glob.glob(pattern)
    if epoch_extractor is None:
        files = sorted(files)
    else:
        decorated = []
        for p in files:
            try:
                decorated.append((int(epoch_extractor(p)), p))
            except Exception:
                continue
        decorated.sort()
        files = [p for _, p in decorated]
    if len(files) <= keep_last_n:
        return
    for p in files[:-keep_last_n]:
        try:
            _os.remove(p)
            if logger is not None:
                logger.info(f"Deleted old file: {p}")
        except OSError:
            pass


def create_logger(logging_dir):
    """
    Create a logger that writes to both stdout and a file.
    If distributed is initialized, only rank 0 will log to the file.
    """
    # Check if distributed is initialized
    is_dist = dist.is_available() and dist.is_initialized() 
    rank    = dist.get_rank() if is_dist else 0             
    
    if rank == 0:
        handlers = [logging.StreamHandler()]                

        if logging_dir is not None:
            os.makedirs(logging_dir, exist_ok=True)
            handlers.append(logging.FileHandler(f"{logging_dir}/log.txt", mode='a')) 
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s", 
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
            force=True # Override any existing logging configuration
        )
        logger = logging.getLogger(__name__)
    else:
        # 非 Rank 0 的日志处理：只创建logger但不记录，节省资源
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler()) # 吞掉所有日志消息，不输出任何内容
        
    return logger




# ---------------------------------------------------------------------------
#  分块联合生成 rollout(README §14 联合版,planb/随机视频预测.md §3.3-B/C)
# ---------------------------------------------------------------------------

class BlockFlowPredictor:
    """分块联合生成包装器：一次 ODE 采样一整块 L 帧,窗口按块滑动。

    块内帧由同一条件分布联合生成；块之间自回归（第 3 维 B + C 的
    分块自回归部署方式）。
    """

    def __init__(self, model, block_len, noise_sampler, sampler_method="heun",
                 ode_steps=8):
        self.model          = model
        self.block_len      = block_len
        self.noise_sampler  = noise_sampler
        self.sampler_method = sampler_method
        self.ode_steps      = ode_steps

    @torch.no_grad()
    def _sample_block(self, context, params, dt, x0=None, reverse=False):
        """对当前历史条件联合采样下一块 [B, L, C, H, W]。"""
        B, K, C, H, W = context.shape
        if x0 is None:
            x0 = self.noise_sampler(context.new_zeros(B, self.block_len * C, H, W))
        block = integrate_ode(
            self.model, x0, params,
            num_steps=self.ode_steps,
            method=self.sampler_method,
            reverse=reverse,
            model_kwargs={"context": context, "params": params, "dt": dt},
        )                                                   # [B, L*C, H, W]
        return block.reshape(B, self.block_len, C, H, W)

    @torch.no_grad()
    def rollout(self, context, num_future_steps, params=None, dt=None,
                forcing=None, geometry_mask=None, num_samples=1):
        """
        Args:
            context: [B, K, C, H, W]

        Returns:
            num_samples=1: [B, T_future, C, H, W]
            num_samples>1: [B, M, T_future, C, H, W]
        """
        B, K, C, H, W   = context.shape
        L               = self.block_len

        def _one_rollout():
            history     = context
            blocks      = []
            n_blocks    = -(-num_future_steps // L)
            for _ in range(n_blocks):
                block   = self._sample_block(history, params, dt)
                blocks.append(block)
                history = torch.cat([history, block], dim=1)[:, -K:]
            return torch.cat(blocks, dim=1)[:, :num_future_steps]

        if num_samples == 1:
            return _one_rollout()
        return torch.stack([_one_rollout() for _ in range(num_samples)], dim=1)


# ---------------------------------------------------------------------------
#  非分块联合生成
# ---------------------------------------------------------------------------
class JointBlockGenerator:
    """仅联合生成(无分块 rollout 展开)：一次 ODE 采样一整块 L 帧。

    与 BlockFlowPredictor 的区别：不包含 rollout/展开循环,只做单次联合生成——
    无时间展开,计算量 = 一次 ODE 积分(O(ode_steps x F(L))),块内帧并行生成。
    """

    def __init__(self, model, block_len, noise_sampler, sampler_method="heun",
                 ode_steps=8):
        self.model          = model
        self.block_len      = block_len
        self.noise_sampler  = noise_sampler
        self.sampler_method = sampler_method
        self.ode_steps      = ode_steps

    @torch.no_grad()
    def generate(self, context, params, dt, x0=None, reverse=False):
        """对当前历史条件联合生成一块 [B, L, C, H, W](单次 ODE,无展开)。"""
        B, K, C, H, W = context.shape
        if x0 is None:
            x0 = self.noise_sampler(context.new_zeros(B, self.block_len * C, H, W))
        block = integrate_ode(
            self.model, x0, params,
            num_steps=self.ode_steps,
            method=self.sampler_method,
            reverse=reverse,
            model_kwargs={"context": context, "params": params, "dt": dt},
        )                                                   # [B, L*C, H, W]
        return block.reshape(B, self.block_len, C, H, W)


    
    @torch.no_grad()
    def rollout_blocks(model, context, num_future_steps, params, dt,
                    sampler_method, ode_steps, noise_sampler, block_len):
        """分块联合生成闭环 rollout:每步 ODE 联合采样 block_len 帧,窗口按块滑动。

        对应训练端 BlockFlowPredictor.rollout(展开循环)。训练侧改为仅联合后,
        本函数暂时注释保留,后续做长程分块对比时取消注释复用。

        Args:
            context: [B, K, C, H, W] 真实初始窗口

        Returns:
            [B, T, C, H, W],T = num_future_steps(不足一块时截断)
        """
        B, K, C, H, W = context.shape
        history = context
        blocks = []
        n_blocks = -(-num_future_steps // block_len)
        for _ in range(n_blocks):
            x0 = noise_sampler(history.new_zeros(B, block_len * C, H, W))
            block = integrate_ode(
                model, x0, params,
                num_steps=ode_steps, method=sampler_method,
                model_kwargs={"context": history, "params": params, "dt": dt},
            )
            block = block.reshape(B, block_len, C, H, W)
            blocks.append(block)
            history = torch.cat([history, block], dim=1)[:, -K:]
        return torch.cat(blocks, dim=1)[:, :num_future_steps]



@torch.no_grad()
def integrate_ode(model, x0, parameters, num_steps=32, reverse=False, method="euler",
                  rtol=1e-5, atol=1e-5, model_kwargs=None):
    """ODE-based FM sampler using torchdiffeq.

    Methods supported: 'euler', 'heun2', 'rk4', 'dopri5'.

    Args:
        model: FM 向量场。默认旧接口调用 ``model(xt, t_batch, parameters)``；
               传入 ``model_kwargs``（如 ``dict(context=..., params=..., dt=...)``）时,
               改为调用 ``model(xt, t_batch, **model_kwargs)``,适配
               theory/README.md §7.1 的统一条件向量场接口。
        x0:     初始状态 [B, ...]。
        parameters: 标量条件（旧接口专用；新接口下经 model_kwargs 传递,此处仅占位）。
    """
    B       = x0.shape[0]
    device  = x0.device

    model.eval()

    def ode_func(t, xt):
        t_batch = torch.full((B, 1), float(t), device=device, dtype=x0.dtype)
        if model_kwargs is not None:
            dx      = model(xt, t_batch, **model_kwargs)
        else:
            dx      = model(xt, t_batch, parameters)
        return dx

    if reverse:
        t_span = torch.linspace(1.0, 0.0, num_steps, device=device, dtype=x0.dtype)
    else:
        t_span = torch.linspace(0.0, 1.0, num_steps, device=device, dtype=x0.dtype)

    trajectory = odeint(ode_func, x0, t_span, method=method, rtol=rtol, atol=atol)
    return trajectory[-1]
