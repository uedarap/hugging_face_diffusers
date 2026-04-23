"""Valida o ambiente local para a POC.

Este script existe principalmente para estudo. Ele tenta responder perguntas como:

- Meu Python está em uma versão razoável?
- As dependências principais estão instaladas?
- O PyTorch está visível?
- O CUDA está disponível?
- Qual device final a POC escolheria?
- Os diretórios de saída existem ou podem ser criados?

O objetivo é tornar o ambiente observável antes de gastar tempo baixando modelos
ou esperando uma inferência falhar.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from pathlib import Path

from src.config import ConfigurationError, build_base_config
from src.profiles import get_profile, list_profiles
from src.utils.device import collect_hardware_summary


REQUIRED_PACKAGES = [
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "PIL",
    "yaml",
    "imageio",
    "dotenv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida o ambiente local da POC Diffusers.")
    parser.add_argument("--config", default="config.yaml", help="Arquivo de configuração YAML.")
    parser.add_argument("--profile", help="Perfil desejado.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Device desejado.")
    return parser.parse_args()


def check_python_version() -> list[str]:
    messages: list[str] = []
    version_info = sys.version_info
    messages.append(f"Python detectado: {platform.python_version()}")
    if version_info < (3, 10):
        messages.append("AVISO: recomenda-se Python 3.10+ para melhor compatibilidade.")
    else:
        messages.append("OK: versão de Python adequada para a POC.")
    return messages


def check_packages() -> list[str]:
    messages: list[str] = []
    for package_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package_name)
            messages.append(f"OK: dependência '{package_name}' encontrada.")
        except ImportError:
            messages.append(f"FALTA: dependência '{package_name}' não encontrada.")
    return messages


def check_output_dirs(base_config) -> list[str]:
    messages: list[str] = []
    for directory in [
        Path(base_config.outputs.images_dir),
        Path(base_config.outputs.videos_dir),
    ]:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            messages.append(f"OK: diretório pronto -> {directory}")
        except OSError as exc:
            messages.append(f"ERRO: não foi possível preparar o diretório {directory}: {exc}")
    return messages


def build_runtime_messages(base_config) -> list[str]:
    messages: list[str] = []
    hardware = collect_hardware_summary(base_config.runtime.device_mode)

    messages.append(f"Device mode solicitado: {base_config.runtime.device_mode}")
    messages.append(f"Device resolvido: {hardware.resolved_device}")
    messages.append(f"CUDA disponível: {hardware.cuda_available}")
    if hardware.gpu_name:
        messages.append(f"GPU detectada: {hardware.gpu_name}")
    else:
        messages.append("GPU detectada: nenhuma")

    if base_config.runtime.device_mode == "cuda" and not hardware.cuda_available:
        messages.append("AVISO: o modo 'cuda' foi solicitado, mas CUDA não está disponível.")

    if hardware.resolved_device == "cpu":
        messages.append("AVISO: em CPU a geração de vídeo pode ser bastante lenta.")
    else:
        messages.append("OK: ambiente com CUDA detectado para acelerar a inferência.")

    return messages


def main() -> int:
    args = parse_args()

    print("=" * 80)
    print("CHECK DE AMBIENTE - POC HUGGING FACE DIFFUSERS")
    print("=" * 80)

    for line in check_python_version():
        print(line)

    print("\n[Dependências]")
    for line in check_packages():
        print(line)

    print("\n[Perfis disponíveis]")
    for profile_name in list_profiles():
        profile = get_profile(profile_name)
        print(f"- {profile.name}: {profile.description}")

    try:
        base_config = build_base_config(
            config_path=args.config,
            cli_overrides={
                "runtime.performance_profile": args.profile,
                "runtime.device_mode": args.device,
            },
        )
    except ConfigurationError as exc:
        print(f"\n[CONFIG ERROR] {exc}")
        return 2

    print("\n[Configuração resolvida]")
    print(f"- profile: {base_config.runtime.performance_profile}")
    print(f"- requested device: {base_config.runtime.device_mode}")
    print(f"- resolved dtype: {base_config.runtime.resolved_dtype}")
    print(f"- image model: {base_config.models.image_model}")
    print(f"- video model: {base_config.models.video_model}")

    print("\n[Hardware]")
    for line in build_runtime_messages(base_config):
        print(f"- {line}")

    print("\n[Diretórios]")
    for line in check_output_dirs(base_config):
        print(f"- {line}")

    print("\n[Sugestões]")
    if base_config.runtime.resolved_device == "cpu":
        print("- Comece pelo perfil 'cpu_safe' e teste imagem antes de vídeo.")
    else:
        print("- Teste primeiro imagem e depois suba os parâmetros de vídeo aos poucos.")

    print("- Se houver erro de memória, reduza resolução, steps e número de frames.")
    print("- Se o modelo exigir autenticação, configure HF_TOKEN no seu .env.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
