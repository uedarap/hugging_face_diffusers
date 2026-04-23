"""Exportação de vídeo a partir de frames PIL."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def build_output_path(output_dir: str, prefix: str, extension: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / f"{prefix}_{timestamp}.{extension}"


def save_video(
    frames,
    output_dir: str,
    explicit_output_path: str | None,
    fps: int,
    prefix: str,
) -> str:
    try:
        if explicit_output_path:
            output_path = Path(explicit_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = build_output_path(output_dir, prefix, "mp4")

        frame_arrays = [np.array(frame) for frame in frames]
        imageio.mimsave(output_path, frame_arrays, fps=fps)
        return str(output_path)
    except OSError as exc:
        raise OSError(f"Falha ao exportar vídeo em '{explicit_output_path or output_dir}': {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"Frames inválidos para exportação de vídeo: {exc}") from exc
