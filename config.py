"""Central configuration for the explainable VQA system."""

from dataclasses import dataclass
from typing import Literal


DeviceMode = Literal["auto", "cuda", "cpu"]
ModelProfile = Literal["vilt", "blip2"]


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model loading and inference."""

    model_profile: ModelProfile = "vilt"
    vqa_model_id: str = "dandelin/vilt-b32-finetuned-vqa"
    clip_model_id: str = "openai/clip-vit-base-patch32"
    device: DeviceMode = "auto"
    use_fp16_on_cuda: bool = True
    max_new_tokens_answer: int = 24
    max_new_tokens_rationale: int = 48
    num_context_turns: int = 4
    temperature: float = 0.0
    do_sample: bool = False

    def get_vqa_model_id(self) -> str:
        """Get the model ID based on the profile if not manually overridden."""
        if self.model_profile == "vilt":
            return "dandelin/vilt-b32-finetuned-vqa"
        if self.model_profile == "blip2":
            return "Salesforce/blip2-opt-2.7b"
        return self.vqa_model_id


@dataclass(frozen=True)
class ExplainabilityConfig:
    """Configuration for visual explanation generation."""

    heatmap_threshold: float = 0.62
    min_box_area_ratio: float = 0.01
    overlay_alpha: float = 0.45
    max_boxes: int = 5
    clip_patch_size: int = 32
    image_size: int = 224


@dataclass(frozen=True)
class AppConfig:
    """Configuration for the Streamlit application."""

    page_title: str = "Explainable VQA"
    page_icon: str = "VQA"
    max_upload_mb: int = 10
