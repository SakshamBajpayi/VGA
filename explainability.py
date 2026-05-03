"""Generate CLIP-based visual and textual explanations for VQA outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from config import ExplainabilityConfig
from model_loader import LoadedModels


Box = Tuple[int, int, int, int]


@dataclass(frozen=True)
class ExplanationResult:
    """Structured explanation output."""

    heatmap: np.ndarray
    boxes: List[Box]
    annotated_image: Image.Image
    textual_explanation: str
    grounding_phrase: str


def _normalize_map(values: np.ndarray) -> np.ndarray:
    """Normalize an array to [0, 1] with numerical safety.

    Args:
        values: Input array.

    Returns:
        Normalized float array.
    """
    if values.size == 0:
        return np.array([], dtype=np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / (max_value - min_value)).astype(np.float32)


def _build_grounding_phrase(question: str, answer: str) -> str:
    """Build the text query used for visual grounding.

    Args:
        question: User question.
        answer: Predicted answer.

    Returns:
        Compact phrase for CLIP patch relevance.
    """
    cleaned_answer = answer.strip()
    if not cleaned_answer:
        # Fallback if answer is empty for some reason
        return f"The object related to the question: {question.strip()}"
    return f"{question.strip()} Answer: {cleaned_answer}"


def compute_clip_patch_heatmap(
    image: Image.Image,
    phrase: str,
    models: LoadedModels,
    config: ExplainabilityConfig,
) -> np.ndarray:
    """Compute a CLIP patch-text relevance heatmap.

    This uses CLIP's ViT patch tokens and compares each projected patch
    representation against the projected text embedding. It is a semantic
    localization proxy, not a causal proof of the answer.

    Args:
        image: Input RGB image.
        phrase: Text phrase to ground.
        models: Loaded CLIP model and processor.
        config: Explainability configuration.

    Returns:
        Heatmap resized to original image size with values in [0, 1].
    """
    rgb = image.convert("RGB")
    inputs = models.clip_processor(
        text=[phrase],
        images=rgb,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(models.device) for key, value in inputs.items()}

    with torch.no_grad():
        # Explicit internal calls to avoid attribute errors on model outputs
        vision_outputs = models.clip_model.vision_model(
            pixel_values=inputs["pixel_values"],
            output_hidden_states=False,
            return_dict=True,
        )

        text_outputs = models.clip_model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )

        # Vision tokens: [CLS] is index 0, patches are 1:
        patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]
        patch_features = models.clip_model.visual_projection(patch_tokens)

        # Text features from pooled output
        text_pooled = text_outputs.pooler_output
        text_features = models.clip_model.text_projection(text_pooled)

        # Normalize features
        patch_features = torch.nn.functional.normalize(patch_features, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

        # Cosine similarity
        scores = torch.matmul(patch_features, text_features.unsqueeze(-1))
        scores = scores.squeeze(0).squeeze(-1).detach().float().cpu().numpy()

    grid_size = int(np.sqrt(scores.shape[0]))
    if grid_size * grid_size != scores.shape[0]:
        raise RuntimeError(
            f"Unexpected CLIP patch count {scores.shape[0]}; cannot form square {grid_size}x{grid_size} grid."
        )

    patch_map = scores.reshape(grid_size, grid_size)
    patch_map = _normalize_map(patch_map)

    width, height = rgb.size
    heatmap = cv2.resize(patch_map, (width, height), interpolation=cv2.INTER_CUBIC)
    return _normalize_map(heatmap)


def extract_boxes_from_heatmap(
    heatmap: np.ndarray,
    config: ExplainabilityConfig,
) -> List[Box]:
    """Extract bounding boxes from a thresholded heatmap.

    Args:
        heatmap: Normalized heatmap in [0, 1].
        config: Explainability configuration.

    Returns:
        Bounding boxes as (x1, y1, x2, y2).
    """
    if heatmap.size == 0:
        return []

    height, width = heatmap.shape[:2]
    binary = (heatmap >= config.heatmap_threshold).astype(np.uint8) * 255

    kernel = np.ones((7, 7), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = config.min_box_area_ratio * float(width * height)
    boxes_with_area: List[Tuple[Box, float]] = []

    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area = float(box_width * box_height)
        if area >= min_area:
            boxes_with_area.append(((x, y, x + box_width, y + box_height), area))

    boxes_with_area.sort(key=lambda item: item[1], reverse=True)
    return [box for box, _ in boxes_with_area[: config.max_boxes]]


def render_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    boxes: Sequence[Box],
    config: ExplainabilityConfig,
) -> Image.Image:
    """Render heatmap and boxes on the image.

    Args:
        image: Input RGB image.
        heatmap: Normalized heatmap.
        boxes: Bounding boxes to draw.
        config: Explainability configuration.

    Returns:
        Annotated PIL image.
    """
    rgb = np.array(image.convert("RGB"))
    if heatmap.size == 0:
        return Image.fromarray(rgb)

    heat_uint8 = np.uint8(255 * _normalize_map(heatmap))
    colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        rgb,
        1.0 - config.overlay_alpha,
        colored,
        config.overlay_alpha,
        0,
    )

    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(
            overlay,
            f"R{idx}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return Image.fromarray(overlay)


def build_textual_explanation(answer: str, boxes: Sequence[Box]) -> str:
    """Build a concise explanation from grounding output.

    Args:
        answer: Predicted answer.
        boxes: Visual regions supporting the answer.

    Returns:
        Textual explanation.
    """
    if not boxes:
        return (
            f"The model answered '{answer}'. The CLIP grounding map did not produce "
            "a confident region, so the visual evidence should be treated as weak."
        )

    region_names = ", ".join(f"R{idx}" for idx in range(1, len(boxes) + 1))
    return (
        f"The model answered '{answer}' and the strongest text-image alignment "
        f"appears in region(s) {region_names}. These regions are post-hoc evidence "
        "for where the answer phrase is visually grounded."
    )


def generate_explanation(
    image: Image.Image,
    question: str,
    answer: str,
    models: LoadedModels,
    config: ExplainabilityConfig,
) -> ExplanationResult:
    """Generate visual and textual explanations for one VQA result.

    Args:
        image: Input image.
        question: User question.
        answer: Model answer.
        models: Loaded models.
        config: Explainability configuration.

    Returns:
        Explanation result.
    """
    phrase = _build_grounding_phrase(question, answer)
    try:
        heatmap = compute_clip_patch_heatmap(image, phrase, models, config)
        boxes = extract_boxes_from_heatmap(heatmap, config)
    except RuntimeError:
        # Fallback if patch grid fails
        heatmap = np.array([], dtype=np.float32)
        boxes = []

    annotated = render_overlay(image, heatmap, boxes, config)
    textual = build_textual_explanation(answer, boxes)

    return ExplanationResult(
        heatmap=heatmap,
        boxes=boxes,
        annotated_image=annotated,
        textual_explanation=textual,
        grounding_phrase=phrase,
    )
