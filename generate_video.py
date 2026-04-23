"""CLI principal para geração de vídeo por prompt."""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import ConfigurationError, build_video_config
from src.utils.logging_utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera um vídeo por prompt usando Hugging Face Diffusers."
    )
    parser.add_argument("--prompt", required=True, help="Prompt de texto para o vídeo.")
    parser.add_argument("--profile", help="Perfil de execução: auto, cpu_safe, gtx1080, high_quality.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Device desejado.")
    parser.add_argument("--config", default="config.yaml", help="Caminho do arquivo YAML de configuração.")
    parser.add_argument("--seed", type=int, help="Seed para reprodutibilidade.")
    parser.add_argument("--output", help="Caminho completo do arquivo de saída.")
    parser.add_argument("--steps", type=int, help="Quantidade de steps de inferência.")
    parser.add_argument("--frames", type=int, help="Quantidade de frames.")
    parser.add_argument("--width", type=int, help="Largura do vídeo.")
    parser.add_argument("--height", type=int, help="Altura do vídeo.")
    parser.add_argument("--guidance-scale", type=float, help="Guidance scale para a geração.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        app_config = build_video_config(
            config_path=args.config,
            prompt=args.prompt,
            cli_overrides={
                "runtime.device_mode": args.device,
                "runtime.performance_profile": args.profile,
                "video.seed": args.seed,
                "video.steps": args.steps,
                "video.num_frames": args.frames,
                "video.width": args.width,
                "video.height": args.height,
                "video.guidance_scale": args.guidance_scale,
                "outputs.video_output_file": args.output,
            },
        )
        configure_logging(app_config.logging)
        logger = logging.getLogger("generate_video")
        try:
            from src.pipelines.video_generation import generate_video
        except ImportError as exc:
            raise RuntimeError(
                "Dependências de geração de vídeo ausentes. Instale os pacotes do requirements.txt e o PyTorch correto para sua máquina."
            ) from exc
        logger.info("Iniciando geração de vídeo.")
        logger.info(
            "Config final: profile=%s | device=%s | dtype=%s | model=%s",
            app_config.runtime.performance_profile,
            app_config.runtime.resolved_device,
            app_config.runtime.resolved_dtype,
            app_config.models.video_model,
        )
        result = generate_video(app_config)
        logger.info("Vídeo salvo em: %s", result.output_path)
        return 0
    except ConfigurationError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"[RUNTIME ERROR] {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # pragma: no cover - última barreira para uma POC didática.
        print(f"[UNEXPECTED ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
