"""Perfis de execucao da POC.

Perfis existem para representar intencoes de uso, e nao apenas conjuntos
aleatorios de numeros. Cada perfil abaixo foi pensado para estudo e pode
ser adaptado depois.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_PROFILE_NAME = "auto"


@dataclass(frozen=True)
class ExecutionProfile:
    name: str
    description: str
    overrides: dict


PROFILES: dict[str, ExecutionProfile] = {
    "auto": ExecutionProfile(
        name="auto",
        description=(
            "Perfil neutro. Mantem valores moderados e deixa a escolha do device por conta da deteccao automatica."
        ),
        overrides={
            "runtime": {
                "device_mode": "auto",
                "performance_profile": "auto",
                "dtype": "auto",
            }
        },
    ),
    "cpu_safe": ExecutionProfile(
        name="cpu_safe",
        description=(
            "Perfil conservador para maquinas sem GPU. Reduz custo para tornar a POC estudavel em CPU."
        ),
        overrides={
            "runtime": {
                "device_mode": "cpu",
                "performance_profile": "cpu_safe",
                "dtype": "float32",
            },
            "image": {
                "width": 384,
                "height": 384,
                "steps": 15,
                "guidance_scale": 7.0,
            },
            "video": {
                "width": 256,
                "height": 144,
                "steps": 10,
                "num_frames": 8,
                "fps": 6,
                "guidance_scale": 6.5,
            },
        },
    ),
    "gtx1080": ExecutionProfile(
        name="gtx1080",
        description=(
            "Perfil para GPU intermediaria/antiga com CUDA. Prioriza equilibrio entre VRAM limitada e usabilidade."
        ),
        overrides={
            "runtime": {
                "device_mode": "cuda",
                "performance_profile": "gtx1080",
                "dtype": "float16",
            },
            "image": {
                "width": 512,
                "height": 512,
                "steps": 20,
                "guidance_scale": 7.5,
            },
            "video": {
                "width": 320,
                "height": 192,
                "steps": 12,
                "num_frames": 12,
                "fps": 8,
                "guidance_scale": 7.0,
            },
        },
    ),
    "gtx1650": ExecutionProfile(
        name="gtx1650",
        description=(
            "Perfil para GTX 1650 e GPUs com 4 GB de VRAM. Usa float32 para evitar instabilidades comuns com float16."
        ),
        overrides={
            "runtime": {
                "device_mode": "cuda",
                "performance_profile": "gtx1650",
                "dtype": "float32",
            },
            "image": {
                "width": 448,
                "height": 448,
                "steps": 16,
                "guidance_scale": 7.0,
            },
            "video": {
                "width": 256,
                "height": 144,
                "steps": 8,
                "num_frames": 6,
                "fps": 6,
                "guidance_scale": 6.5,
            },
        },
    ),
    "high_quality": ExecutionProfile(
        name="high_quality",
        description=(
            "Perfil para hardware mais forte. Aumenta resolucao e qualidade, aceitando maior custo computacional."
        ),
        overrides={
            "runtime": {
                "device_mode": "cuda",
                "performance_profile": "high_quality",
                "dtype": "float16",
            },
            "image": {
                "width": 768,
                "height": 768,
                "steps": 30,
                "guidance_scale": 8.0,
            },
            "video": {
                "width": 576,
                "height": 320,
                "steps": 20,
                "num_frames": 16,
                "fps": 8,
                "guidance_scale": 7.5,
            },
        },
    ),
}


def get_profile(profile_name: str) -> ExecutionProfile:
    return PROFILES[profile_name]


def list_profiles() -> list[str]:
    return list(PROFILES.keys())


def profile_exists(profile_name: str) -> bool:
    return profile_name in PROFILES
