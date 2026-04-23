"""Pipeline de geração de imagem.

Este módulo encapsula tudo que é específico de image generation.
Ele isola a biblioteca Diffusers do restante da aplicação.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
from diffusers import StableDiffusionPipeline

from src.config import AppConfig
from src.utils.device import build_torch_dtype, validate_device_request
from src.utils.image_export import save_image


logger = logging.getLogger(__name__)


@dataclass
class ImageGenerationResult:
    output_path: str


def _build_generator(device: str, seed: int) -> torch.Generator:
    # O gerador com seed ajuda a tornar resultados mais reproduzíveis.
    # Em CPU e CUDA o construtor de Generator recebe o device correspondente.
    return torch.Generator(device=device).manual_seed(seed)


def generate_image(app_config: AppConfig) -> ImageGenerationResult:
    validate_device_request(app_config.runtime.device_mode, app_config.runtime.resolved_device)

    try:
        torch_dtype = build_torch_dtype(app_config.runtime.resolved_dtype)
        logger.info("Carregando pipeline de imagem: %s", app_config.models.image_model)
        pipeline = StableDiffusionPipeline.from_pretrained(
            app_config.models.image_model,
            torch_dtype=torch_dtype,
        )

        # Attention slicing é um ajuste simples e didático que costuma ajudar
        # em ambientes mais limitados, principalmente quando a memória aperta.
        pipeline.enable_attention_slicing()
        pipeline = pipeline.to(app_config.runtime.resolved_device)

        result = pipeline(
            prompt=app_config.image.prompt,
            height=app_config.image.height,
            width=app_config.image.width,
            num_inference_steps=app_config.image.steps,
            guidance_scale=app_config.image.guidance_scale,
            generator=_build_generator(app_config.runtime.resolved_device, app_config.image.seed),
        )
        image = result.images[0]
        output_path = save_image(
            image=image,
            output_dir=app_config.outputs.images_dir,
            explicit_output_path=app_config.outputs.image_output_file,
            prefix="image",
        )
        return ImageGenerationResult(output_path=output_path)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "Memória insuficiente na GPU durante geração de imagem. Tente reduzir resolução ou steps."
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Falha ao carregar/salvar recursos da geração de imagem: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Falha na geração de imagem: {exc}") from exc
