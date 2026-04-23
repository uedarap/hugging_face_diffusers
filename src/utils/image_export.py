"""Exportação de imagem."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PIL import Image


def build_output_path(output_dir: str, prefix: str, extension: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / f"{prefix}_{timestamp}.{extension}"


def save_image(image: Image.Image, output_dir: str, explicit_output_path: str | None, prefix: str) -> str:
    try:
        if explicit_output_path:
            output_path = Path(explicit_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = build_output_path(output_dir, prefix, "png")
        image.save(output_path)
        return str(output_path)
    except OSError as exc:
        raise OSError(f"Falha ao salvar imagem em '{explicit_output_path or output_dir}': {exc}") from exc
