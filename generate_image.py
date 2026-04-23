"""CLI principal para geracao de imagem por prompt.

Este script e propositalmente fino: ele lida com argumentos, resolve a
configuracao final e delega a inferencia ao modulo de pipeline.
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import ConfigurationError, build_image_config
from src.utils.logging_utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera uma imagem por prompt usando Hugging Face Diffusers."
    )
    parser.add_argument("--prompt", required=True, help="Prompt de texto para a imagem.")
    parser.add_argument(
        "--profile",
        help="Perfil de execucao: auto, cpu_safe, gtx1080, gtx1650, high_quality.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Device desejado.")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16"],
        help="Precisao numerica desejada para o pipeline.",
    )
    parser.add_argument("--config", default="config.yaml", help="Caminho do arquivo YAML de configuracao.")
    parser.add_argument("--seed", type=int, help="Seed para reprodutibilidade.")
    parser.add_argument("--output", help="Caminho completo do arquivo de saida.")
    parser.add_argument("--steps", type=int, help="Quantidade de steps de inferencia.")
    parser.add_argument("--width", type=int, help="Largura da imagem.")
    parser.add_argument("--height", type=int, help="Altura da imagem.")
    parser.add_argument("--guidance-scale", type=float, help="Guidance scale para a geracao.")
    parser.add_argument(
        "--disable-safety-checker",
        action="store_true",
        help="Desativa o safety checker do modelo para testes locais.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        app_config = build_image_config(
            config_path=args.config,
            prompt=args.prompt,
            cli_overrides={
                "runtime.device_mode": args.device,
                "runtime.performance_profile": args.profile,
                "runtime.dtype": args.dtype,
                "image.seed": args.seed,
                "image.steps": args.steps,
                "image.width": args.width,
                "image.height": args.height,
                "image.guidance_scale": args.guidance_scale,
                "outputs.image_output_file": args.output,
            },
        )
        configure_logging(app_config.logging)
        logger = logging.getLogger("generate_image")
        try:
            from src.pipelines.image_generation import generate_image
        except ImportError as exc:
            raise RuntimeError(
                "Dependencias de geracao de imagem ausentes. Instale os pacotes do requirements.txt e o PyTorch correto para sua maquina."
            ) from exc
        logger.info("Iniciando geracao de imagem.")
        logger.info(
            "Config final: profile=%s | device=%s | dtype=%s | model=%s | safety_checker=%s",
            app_config.runtime.performance_profile,
            app_config.runtime.resolved_device,
            app_config.runtime.resolved_dtype,
            app_config.models.image_model,
            "off" if args.disable_safety_checker else "on",
        )
        result = generate_image(
            app_config,
            disable_safety_checker=args.disable_safety_checker,
        )
        logger.info("Imagem salva em: %s", result.output_path)
        return 0
    except ConfigurationError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"[RUNTIME ERROR] {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # pragma: no cover - util como ultima protecao em POC didatica.
        print(f"[UNEXPECTED ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
