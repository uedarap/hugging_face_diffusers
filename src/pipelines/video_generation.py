"""Pipeline de geração de vídeo.

Vídeo é mais pesado que imagem, então este módulo destaca explicitamente alguns
trade-offs e mensagens de erro para ajudar no estudo.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
from diffusers import DiffusionPipeline

from src.config import AppConfig
from src.utils.device import build_torch_dtype, validate_device_request
from src.utils.video_export import save_video


logger = logging.getLogger(__name__)


@dataclass
class VideoGenerationResult:
    output_path: str


def _build_generator(device: str, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def generate_video(app_config: AppConfig) -> VideoGenerationResult:
    validate_device_request(app_config.runtime.device_mode, app_config.runtime.resolved_device)

    try:
        torch_dtype = build_torch_dtype(app_config.runtime.resolved_dtype)
        logger.info("Carregando pipeline de vídeo: %s", app_config.models.video_model)
        pipeline = DiffusionPipeline.from_pretrained(
            app_config.models.video_model,
            torch_dtype=torch_dtype,
        )
        pipeline.enable_attention_slicing()
        pipeline = pipeline.to(app_config.runtime.resolved_device)

        result = pipeline(
            prompt=app_config.video.prompt,
            height=app_config.video.height,
            width=app_config.video.width,
            num_inference_steps=app_config.video.steps,
            guidance_scale=app_config.video.guidance_scale,
            num_frames=app_config.video.num_frames,
            generator=_build_generator(app_config.runtime.resolved_device, app_config.video.seed),
        )

        # Alguns pipelines retornam frames em result.frames[0], outros usam listas
        # ligeiramente diferentes. Esta POC assume o formato mais comum do Diffusers
        # para text-to-video. Se você trocar o modelo, este é um dos pontos a revisar.
        frames = result.frames[0]
        output_path = save_video(
            frames=frames,
            output_dir=app_config.outputs.videos_dir,
            explicit_output_path=app_config.outputs.video_output_file,
            fps=app_config.video.fps,
            prefix="video",
        )
        return VideoGenerationResult(output_path=output_path)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "Memória insuficiente na GPU durante geração de vídeo. Reduza resolução, steps ou frames."
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Falha ao carregar/salvar recursos da geração de vídeo: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Falha na geração de vídeo: {exc}") from exc
