"""Pick the torch device for the detector (serving) and train_detector.py (training).

One rule: use a GPU whenever torch can use one, and say loudly why when it
does not. The request ('' / 'auto' by default, from DETECTOR_DEVICE or
``--device``) resolves like this:

    auto        CUDA or ROCm GPU 0 if torch.cuda.is_available(), else Apple MPS,
                else CPU. ROCm builds of torch answer through torch.cuda too
                (torch.version.hip tells them apart).
    cpu         CPU, on purpose.
    0, 1, cuda, cuda:1, gpu
                that CUDA/ROCm GPU; an error if torch cannot see it.
    mps         Apple GPU; an error if it is not available.

When auto lands on the CPU, the result says why, and flags the case that
matters: GPU hardware on the machine (nvidia-smi, /dev/nvidia*, /dev/kfd,
rocm-smi) that torch cannot use. That is a CPU-only torch build, a missing
driver, CUDA_VISIBLE_DEVICES hiding it, or a container started without
``--gpus all``, and it costs the whole latency budget.
"""

import os
import shutil
from dataclasses import asdict, dataclass, field
from typing import List, Optional

_GPU_ALIASES = ('cuda', 'gpu')


@dataclass
class DeviceChoice:
    device: str                     # what ultralytics gets: 'cpu', '0', '1', 'mps'
    kind: str                       # 'cuda' | 'rocm' | 'mps' | 'cpu'
    how: str                        # 'auto' | 'forced' | 'fallback'
    reason: str
    requested: str = ''
    gpu_name: Optional[str] = None
    torch_version: Optional[str] = None
    torch_build: Optional[str] = None           # 'cuda 12.4' | 'rocm 7.2' | 'cpu-only'
    gpu_hardware_seen: List[str] = field(default_factory=list)
    gpu_hardware_unused: bool = False           # hardware present, torch not using it

    @property
    def is_gpu(self) -> bool:
        return self.kind != 'cpu'

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        where = self.kind.upper() + (f' ({self.gpu_name})' if self.gpu_name else '')
        return f'device {self.device!r} = {where}, {self.how}: {self.reason}'


def gpu_hardware_hints() -> List[str]:
    """Signs of GPU hardware on this machine, whatever torch thinks."""
    hints = []
    for tool in ('nvidia-smi', 'rocm-smi', 'amd-smi'):
        if shutil.which(tool):
            hints.append(f'{tool} on PATH')
    for path in ('/dev/nvidia0', '/dev/nvidiactl', '/dev/kfd'):
        if os.path.exists(path):
            hints.append(path)
    return hints


def _torch_build(torch) -> str:
    version = getattr(torch, 'version', None)
    if getattr(version, 'hip', None):
        return f'rocm {version.hip}'
    if getattr(version, 'cuda', None):
        return f'cuda {version.cuda}'
    return 'cpu-only'


def _hidden_by_env() -> Optional[str]:
    for name in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        value = os.environ.get(name)
        if value is not None and value.strip() in ('', '-1', 'none', 'NoDevFiles'):
            return f'{name}={value!r} hides every GPU'
    return None


def _gpu_index(requested: str) -> Optional[int]:
    """'0' -> 0, 'cuda:1' -> 1, 'cuda'/'gpu' -> 0, anything else -> None."""
    value = requested.lower()
    if value in _GPU_ALIASES:
        return 0
    if value.startswith('cuda:'):
        value = value[5:]
    return int(value) if value.isdigit() else None


def select_device(requested: str = '') -> DeviceChoice:
    requested = (requested or '').strip()
    wanted = requested.lower()
    hints = gpu_hardware_hints()

    try:
        import torch
    except Exception as error:   # torch missing or broken: only the CPU is left, say so
        if wanted not in ('', 'auto', 'cpu'):
            raise RuntimeError(f'Device {requested!r} requested but torch failed to import: {error}') from error
        return DeviceChoice('cpu', 'cpu', 'fallback', f'torch failed to import ({error})', requested,
                            gpu_hardware_seen=hints, gpu_hardware_unused=bool(hints))

    base = dict(requested=requested, torch_version=torch.__version__, torch_build=_torch_build(torch),
                gpu_hardware_seen=hints)
    cuda_error = None
    try:
        cuda_ok = bool(torch.cuda.is_available())
    except Exception as error:   # broken driver / runtime: no GPU, but say why
        cuda_ok, cuda_error = False, f'torch.cuda.is_available() raised {type(error).__name__}: {error}'
    gpu_kind = 'rocm' if getattr(torch.version, 'hip', None) else 'cuda'
    mps = getattr(torch.backends, 'mps', None)
    mps_ok = bool(mps) and _safe(mps.is_available)

    if wanted == 'cpu':
        return DeviceChoice('cpu', 'cpu', 'forced', 'CPU requested explicitly',
                            gpu_hardware_unused=cuda_ok or mps_ok, **base)
    if wanted == 'mps':
        if not mps_ok:
            raise RuntimeError('Device mps requested but torch.backends.mps.is_available() is False')
        return DeviceChoice('mps', 'mps', 'forced', 'MPS requested explicitly', gpu_name='Apple GPU', **base)
    index = _gpu_index(wanted)
    if index is not None:
        if not cuda_ok:
            raise RuntimeError(
                f'GPU {requested!r} requested but torch.cuda.is_available() is False '
                f'(torch {torch.__version__}, {_torch_build(torch)}; {_why_no_gpu(torch, hints, cuda_error)})'
            )
        count = _safe(torch.cuda.device_count, 0)
        if index >= count:
            raise RuntimeError(f'GPU {requested!r} requested but torch sees only {count} GPU(s)')
        return DeviceChoice(str(index), gpu_kind, 'forced', f'GPU {index} requested explicitly',
                            gpu_name=_safe(lambda: torch.cuda.get_device_name(index)), **base)
    if wanted not in ('', 'auto'):
        raise RuntimeError(f"Unknown device {requested!r}: use auto, cpu, 0, cuda:N or mps")

    if cuda_ok:
        return DeviceChoice('0', gpu_kind, 'auto', f'{gpu_kind.upper()} GPU available to torch',
                            gpu_name=_safe(lambda: torch.cuda.get_device_name(0)), **base)
    if mps_ok:
        return DeviceChoice('mps', 'mps', 'auto', 'Apple MPS available to torch', gpu_name='Apple GPU', **base)
    return DeviceChoice('cpu', 'cpu', 'auto', f'no GPU usable by torch: {_why_no_gpu(torch, hints, cuda_error)}',
                        gpu_hardware_unused=bool(hints), **base)


def _why_no_gpu(torch, hints, cuda_error=None) -> str:
    reasons = [cuda_error] if cuda_error else []
    hidden = _hidden_by_env()
    if hidden:
        reasons.append(hidden)
    if _torch_build(torch) == 'cpu-only':
        reasons.append(f'torch {torch.__version__} is a CPU-only build')
    if hints:
        reasons.append('but GPU hardware is present (' + ', '.join(hints) + ')')
    elif not reasons:
        reasons.append('no GPU hardware found either')
    return '; '.join(reasons)


def _safe(fn, default=False):
    try:
        return fn()
    except Exception:
        return default
