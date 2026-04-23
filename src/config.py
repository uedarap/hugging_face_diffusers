"""Resolução central de configuração.

Este módulo é o coração organizacional da POC.

Ele pega várias fontes de configuração e transforma tudo em um objeto final:

1. YAML base
2. variáveis de ambiente
3. perfil selecionado
4. overrides do CLI

Depois disso, ele resolve device e dtype finais e valida os valores.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import yaml

from src.profiles import DEFAULT_PROFILE_NAME, get_profile, profile_exists
from src.utils.device import resolve_device_and_dtype


class ConfigurationError(ValueError):
    """Erro dedicado para deixar falhas de configuração mais fáceis de entender."""


@dataclass
class RuntimeConfig:
    device_mode: str
    performance_profile: str
    dtype: str
    resolved_device: str
    resolved_dtype: str


@dataclass
class ModelsConfig:
    image_model: str
    video_model: str


@dataclass
class ImageConfig:
    prompt: str | None
    width: int
    height: int
    steps: int
    guidance_scale: float
    seed: int


@dataclass
class VideoConfig:
    prompt: str | None
    width: int
    height: int
    steps: int
    guidance_scale: float
    num_frames: int
    fps: int
    seed: int


@dataclass
class OutputsConfig:
    images_dir: str
    videos_dir: str
    image_output_file: str | None
    video_output_file: str | None


@dataclass
class LoggingConfig:
    level: str
    format: str


@dataclass
class AppConfig:
    runtime: RuntimeConfig
    models: ModelsConfig
    image: ImageConfig
    video: VideoConfig
    outputs: OutputsConfig
    logging: LoggingConfig


def load_yaml_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise ConfigurationError(f"Arquivo de configuração não encontrado: {config_path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Falha ao ler YAML: {exc}") from exc


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def set_nested_value(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    target = data
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            target[key] = {}
        target = target[key]
    target[keys[-1]] = value


def read_env_overrides() -> dict[str, Any]:
    load_dotenv()
    env_map = {
        "DEVICE_MODE": "runtime.device_mode",
        "PERFORMANCE_PROFILE": "runtime.performance_profile",
        "LOG_LEVEL": "logging.level",
        "IMAGES_OUTPUT_DIR": "outputs.images_dir",
        "VIDEOS_OUTPUT_DIR": "outputs.videos_dir",
    }
    result: dict[str, Any] = {}
    for env_name, dotted_key in env_map.items():
        env_value = os.getenv(env_name)
        if env_value:
            set_nested_value(result, dotted_key, env_value)
    return result


def normalize_cli_overrides(cli_overrides: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not cli_overrides:
        return result
    for dotted_key, value in cli_overrides.items():
        if value is not None:
            set_nested_value(result, dotted_key, value)
    return result


def validate_config(data: dict[str, Any]) -> None:
    device_mode = data["runtime"]["device_mode"]
    if device_mode not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError(f"device_mode inválido: {device_mode}")

    profile_name = data["runtime"]["performance_profile"]
    if not profile_exists(profile_name):
        raise ConfigurationError(f"Perfil inválido: {profile_name}")

    dtype = data["runtime"]["dtype"]
    if dtype not in {"auto", "float32", "float16"}:
        raise ConfigurationError(f"dtype inválido: {dtype}")

    for field_name in ["width", "height", "steps", "seed"]:
        if data["image"][field_name] <= 0:
            raise ConfigurationError(f"Valor inválido em image.{field_name}")

    for field_name in ["width", "height", "steps", "num_frames", "fps", "seed"]:
        if data["video"][field_name] <= 0:
            raise ConfigurationError(f"Valor inválido em video.{field_name}")

    if data["image"]["guidance_scale"] <= 0:
        raise ConfigurationError("image.guidance_scale deve ser > 0")
    if data["video"]["guidance_scale"] <= 0:
        raise ConfigurationError("video.guidance_scale deve ser > 0")

    if not data["models"]["image_model"]:
        raise ConfigurationError("models.image_model não pode ser vazio")
    if not data["models"]["video_model"]:
        raise ConfigurationError("models.video_model não pode ser vazio")


def build_base_config(config_path: str, cli_overrides: dict[str, Any] | None = None) -> AppConfig:
    yaml_config = load_yaml_config(config_path)
    env_overrides = read_env_overrides()
    cli_override_dict = normalize_cli_overrides(cli_overrides)

    merged = deep_merge_dict(yaml_config, env_overrides)

    profile_name = (
        cli_override_dict.get("runtime", {}).get("performance_profile")
        or env_overrides.get("runtime", {}).get("performance_profile")
        or yaml_config.get("runtime", {}).get("performance_profile")
        or DEFAULT_PROFILE_NAME
    )

    if not profile_exists(profile_name):
        raise ConfigurationError(f"Perfil inválido: {profile_name}")

    profile = get_profile(profile_name)
    merged = deep_merge_dict(merged, profile.overrides)
    merged = deep_merge_dict(merged, cli_override_dict)

    validate_config(merged)

    resolved_device, resolved_dtype = resolve_device_and_dtype(
        requested_device=merged["runtime"]["device_mode"],
        requested_dtype=merged["runtime"]["dtype"],
    )

    return AppConfig(
        runtime=RuntimeConfig(
            device_mode=merged["runtime"]["device_mode"],
            performance_profile=merged["runtime"]["performance_profile"],
            dtype=merged["runtime"]["dtype"],
            resolved_device=resolved_device,
            resolved_dtype=resolved_dtype,
        ),
        models=ModelsConfig(
            image_model=merged["models"]["image_model"],
            video_model=merged["models"]["video_model"],
        ),
        image=ImageConfig(
            prompt=None,
            width=merged["image"]["width"],
            height=merged["image"]["height"],
            steps=merged["image"]["steps"],
            guidance_scale=float(merged["image"]["guidance_scale"]),
            seed=merged["image"]["seed"],
        ),
        video=VideoConfig(
            prompt=None,
            width=merged["video"]["width"],
            height=merged["video"]["height"],
            steps=merged["video"]["steps"],
            guidance_scale=float(merged["video"]["guidance_scale"]),
            num_frames=merged["video"]["num_frames"],
            fps=merged["video"]["fps"],
            seed=merged["video"]["seed"],
        ),
        outputs=OutputsConfig(
            images_dir=merged["outputs"]["images_dir"],
            videos_dir=merged["outputs"]["videos_dir"],
            image_output_file=merged.get("outputs", {}).get("image_output_file"),
            video_output_file=merged.get("outputs", {}).get("video_output_file"),
        ),
        logging=LoggingConfig(
            level=merged["logging"]["level"],
            format=merged["logging"]["format"],
        ),
    )


def build_image_config(
    config_path: str,
    prompt: str,
    cli_overrides: dict[str, Any] | None = None,
) -> AppConfig:
    app_config = build_base_config(config_path, cli_overrides)
    app_config.image.prompt = prompt
    return app_config


def build_video_config(
    config_path: str,
    prompt: str,
    cli_overrides: dict[str, Any] | None = None,
) -> AppConfig:
    app_config = build_base_config(config_path, cli_overrides)
    app_config.video.prompt = prompt
    return app_config
