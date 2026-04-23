"""Utilitários relacionados a hardware, device e dtype.

Este módulo tenta ser tolerante com ambientes ainda incompletos. Isso é útil
para o `check_environment.py`, que deve conseguir rodar e emitir diagnósticos
mesmo quando o PyTorch ainda não estiver instalado.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover - depende do ambiente do usuário.
    torch = None


@dataclass
class HardwareSummary:
    requested_device: str
    resolved_device: str
    resolved_dtype: str
    cuda_available: bool
    gpu_name: str | None


def resolve_device_and_dtype(requested_device: str, requested_dtype: str) -> tuple[str, str]:
    cuda_available = bool(torch and torch.cuda.is_available())

    if requested_device == "auto":
        resolved_device = "cuda" if cuda_available else "cpu"
    elif requested_device == "cuda":
        resolved_device = "cuda" if cuda_available else "cpu"
    else:
        resolved_device = "cpu"

    if requested_dtype == "auto":
        resolved_dtype = "float16" if resolved_device == "cuda" else "float32"
    else:
        resolved_dtype = requested_dtype

    if resolved_device == "cpu" and resolved_dtype == "float16":
        # Em CPU, float16 pode causar incompatibilidades ou desempenho ruim.
        resolved_dtype = "float32"

    return resolved_device, resolved_dtype


def build_torch_dtype(dtype_name: str):
    if torch is None:
        raise RuntimeError(
            "PyTorch não está instalado. Instale 'torch' antes de executar a geração."
        )
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


def collect_hardware_summary(requested_device: str) -> HardwareSummary:
    resolved_device, resolved_dtype = resolve_device_and_dtype(requested_device, "auto")
    cuda_available = bool(torch and torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    return HardwareSummary(
        requested_device=requested_device,
        resolved_device=resolved_device,
        resolved_dtype=resolved_dtype,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
    )


def validate_device_request(requested_device: str, resolved_device: str) -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch não está instalado. Rode 'pip install torch ...' antes de gerar imagem ou vídeo."
        )
    if requested_device == "cuda" and resolved_device != "cuda":
        raise RuntimeError(
            "O modo 'cuda' foi solicitado, mas CUDA não está disponível neste ambiente."
        )
