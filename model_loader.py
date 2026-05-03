"""Load and cache pretrained VQA and CLIP models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Any

import torch
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    ViltForQuestionAnswering,
    ViltProcessor,
    CLIPModel,
    CLIPProcessor,
)

from config import ModelConfig, ModelProfile


@dataclass(frozen=True)
class LoadedModels:
    """Container for all pretrained models used by the system."""

    model_profile: ModelProfile
    vqa_processor: Any
    vqa_model: Any
    clip_processor: CLIPProcessor
    clip_model: CLIPModel
    device: torch.device
    dtype: torch.dtype


def resolve_device(device_mode: str) -> torch.device:
    """Resolve requested device mode to a concrete torch device.

    Args:
        device_mode: One of "auto", "cuda", or "cpu".

    Returns:
        Torch device.

    Raises:
        ValueError: If the requested device is invalid or unavailable.
    """
    if device_mode == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_mode == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_mode == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device mode: {device_mode}")


def infer_dtype(device: torch.device, use_fp16_on_cuda: bool) -> torch.dtype:
    """Choose inference dtype based on hardware.

    Args:
        device: Target torch device.
        use_fp16_on_cuda: Whether to use float16 on CUDA.

    Returns:
        Torch dtype for model weights.
    """
    if device.type == "cuda" and use_fp16_on_cuda:
        return torch.float16
    return torch.float32


@lru_cache(maxsize=2)
def load_models_cached(
    model_profile: ModelProfile,
    vqa_model_id: str,
    clip_model_id: str,
    device_mode: str,
    use_fp16_on_cuda: bool,
) -> LoadedModels:
    """Load models with process-level caching.

    Args:
        model_profile: The VQA model profile to load.
        vqa_model_id: Hugging Face VQA checkpoint.
        clip_model_id: Hugging Face CLIP checkpoint.
        device_mode: Device selection mode.
        use_fp16_on_cuda: Whether to load models in fp16 on CUDA.

    Returns:
        Loaded model bundle.

    Raises:
        RuntimeError: If a model cannot be loaded.
        ValueError: If an unsupported profile is provided.
    """
    device = resolve_device(device_mode)
    dtype = infer_dtype(device, use_fp16_on_cuda)

    try:
        if model_profile == "vilt":
            vqa_processor = ViltProcessor.from_pretrained(vqa_model_id)
            vqa_model = ViltForQuestionAnswering.from_pretrained(
                vqa_model_id,
                torch_dtype=dtype,
            )
        elif model_profile == "blip2":
            vqa_processor = Blip2Processor.from_pretrained(vqa_model_id)
            vqa_model = Blip2ForConditionalGeneration.from_pretrained(
                vqa_model_id,
                torch_dtype=dtype,
            )
        else:
            raise ValueError(f"Unsupported model profile: {model_profile}")

        vqa_model.to(device)
        vqa_model.eval()

        clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        clip_model = CLIPModel.from_pretrained(clip_model_id)
        clip_model.to(device)
        clip_model.eval()
    except OSError as exc:
        raise RuntimeError(
            f"Failed to load Hugging Face checkpoints for {model_profile}. "
            "Check internet access, model IDs, and local cache."
        ) from exc

    return LoadedModels(
        model_profile=model_profile,
        vqa_processor=vqa_processor,
        vqa_model=vqa_model,
        clip_processor=clip_processor,
        clip_model=clip_model,
        device=device,
        dtype=dtype,
    )


def load_models(config: Optional[ModelConfig] = None) -> LoadedModels:
    """Load configured models.

    Args:
        config: Optional model configuration.

    Returns:
        Loaded model bundle.
    """
    cfg = config or ModelConfig()
    return load_models_cached(
        cfg.model_profile,
        cfg.get_vqa_model_id(),
        cfg.clip_model_id,
        cfg.device,
        cfg.use_fp16_on_cuda,
    )
