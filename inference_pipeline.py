"""End-to-end multi-turn VQA inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from PIL import Image

from config import ExplainabilityConfig, ModelConfig
from explainability import ExplanationResult, generate_explanation
from model_loader import LoadedModels, load_models


@dataclass(frozen=True)
class ConversationTurn:
    """One VQA conversation turn."""

    question: str
    answer: str


@dataclass
class ConversationState:
    """Conversation memory for one uploaded image."""

    turns: List[ConversationTurn] = field(default_factory=list)

    def add_turn(self, question: str, answer: str) -> None:
        """Add a completed conversation turn.

        Args:
            question: User question.
            answer: Model answer.
        """
        self.turns.append(ConversationTurn(question=question, answer=answer))

    def recent_context(self, max_turns: int) -> str:
        """Serialize recent turns for prompting.

        Args:
            max_turns: Number of recent turns to keep.

        Returns:
            Compact context string.
        """
        selected = self.turns[-max_turns:]
        if not selected:
            return ""
        lines = []
        for turn in selected:
            lines.append(f"Q: {turn.question}\nA: {turn.answer}")
        return "\n".join(lines)


@dataclass(frozen=True)
class VQAResult:
    """Structured VQA result."""

    answer: str
    explanation: ExplanationResult
    prompt: str


class VQAInferencePipeline:
    """Owns VQA prompting, generation, and explanation."""

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        explainability_config: Optional[ExplainabilityConfig] = None,
        models: Optional[LoadedModels] = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            model_config: Model configuration.
            explainability_config: Explanation configuration.
            models: Optional preloaded models.
        """
        self.model_config = model_config or ModelConfig()
        self.explainability_config = explainability_config or ExplainabilityConfig()
        self.models = models or load_models(self.model_config)

    def _build_prompt(self, question: str, state: ConversationState) -> str:
        """Build a compact BLIP-2 prompt.

        Args:
            question: Current user question.
            state: Conversation state.

        Returns:
            Prompt string.
        """
        if self.models.model_profile == "vilt":
            # ViLT is not conversational, use only current question
            return question.strip()

        context = state.recent_context(self.model_config.num_context_turns)
        if context:
            return (
                "Answer the current visual question using the image and the "
                "conversation context. Keep the answer short.\n"
                f"{context}\nCurrent Q: {question}\nA:"
            )
        return f"Question: {question} Answer:"

    def _generate_answer_blip2(
        self,
        image: Image.Image,
        prompt: str,
    ) -> str:
        """Generate text from BLIP-2."""
        inputs = self.models.vqa_processor(
            images=image.convert("RGB"),
            text=prompt,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.models.device) for key, value in inputs.items()}

        autocast_enabled = (
            self.models.device.type == "cuda"
            and self.model_config.use_fp16_on_cuda
        )

        with torch.no_grad():
            with torch.autocast(
                device_type=self.models.device.type,
                enabled=autocast_enabled,
            ):
                generated = self.models.vqa_model.generate(
                    **inputs,
                    max_new_tokens=self.model_config.max_new_tokens_answer,
                    do_sample=self.model_config.do_sample,
                    temperature=None if not self.model_config.do_sample else self.model_config.temperature,
                )

        text = self.models.vqa_processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0]
        return text.strip()

    def _predict_answer_vilt(
        self,
        image: Image.Image,
        question: str,
    ) -> str:
        """Run classification and decode argmax for ViLT."""
        inputs = self.models.vqa_processor(
            images=image.convert("RGB"),
            text=question,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.models.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.models.vqa_model(**inputs)
            logits = outputs.logits
            idx = torch.argmax(logits, dim=-1).item()
            answer = self.models.vqa_model.config.id2label[idx]
        return answer

    def answer_question(
        self,
        image: Image.Image,
        question: str,
        state: ConversationState,
    ) -> VQAResult:
        """Answer a visual question and generate explanations.

        Args:
            image: Input image.
            question: User question.
            state: Conversation state for the current image.

        Returns:
            VQA result.

        Raises:
            ValueError: If question is empty.
        """
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question cannot be empty.")

        prompt = self._build_prompt(cleaned_question, state)

        if self.models.model_profile == "vilt":
            answer = self._predict_answer_vilt(image, cleaned_question)
        else:
            answer = self._generate_answer_blip2(image, prompt)

        explanation = generate_explanation(
            image=image,
            question=cleaned_question,
            answer=answer,
            models=self.models,
            config=self.explainability_config,
        )

        state.add_turn(cleaned_question, answer)

        return VQAResult(
            answer=answer,
            explanation=explanation,
            prompt=prompt,
        )
