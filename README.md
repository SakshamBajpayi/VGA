# Explainable Visual Question Answering (VQA) Dashboard

## Abstract
This project implements a modular Visual Question Answering system with a focus on post-hoc interpretability. By combining state-of-the-art vision-language models (ViLT, BLIP-2) with CLIP-based visual grounding, the dashboard allows users to not only receive answers to natural language questions about images but also visualize the semantic alignment between the model's output and specific image regions.

## Model Profiles
The system supports two distinct architectural approaches to VQA:

| Profile | Model ID | Description | Hardware |
| :--- | :--- | :--- | :--- |
| **ViLT** | `vilt-b32-finetuned-vqa` | Classification-based, optimized for speed. | CPU Friendly |
| **BLIP-2** | `blip2-opt-2.7b` | Generative, supports multi-turn context. | 8GB+ RAM / GPU |

## Explanation Method: Semantic Grounding
Interpretability is achieved using **CLIP ViT Patch-Text Relevance**. 

1. **Answer Context:** The user's question and the model's predicted answer are combined into a single grounding phrase.
2. **Feature Extraction:** CLIP's Vision Transformer (ViT) extracts patch-level visual tokens, while the Text Encoder embeds the grounding phrase.
3. **Similarity Mapping:** We compute the cosine similarity between the textual embedding and every visual patch embedding.
4. **Visualization:** The resulting similarity scores are normalized into a heatmap, and significant regions are extracted as bounding boxes using OpenCV contours.

### Causal vs. Semantic Interpretability
It is critical to note that this method provides **post-hoc semantic evidence**. It demonstrates where a separate model (CLIP) finds alignment between the answer text and image regions. It is **not** a causal trace of the internal weights of the VQA model itself.

## Limitations
- **Grounding Fidelity:** High semantic similarity does not guarantee the VQA model "saw" that region.
- **VQA Reasoning:** ViLT may struggle with complex spatial relations; BLIP-2 on CPU has significant latency.
- **Resource Usage:** Large model downloads (up to several GBs) occur on the first run.

## Setup & Usage
1. **Environment:** `python -m venv .venv` and activate it.
2. **Dependencies:** `pip install -r requirements.txt`
3. **Run:** `streamlit run app.py`

## Portfolio Highlights
- **Architecture:** Demonstrates integration of multiple Hugging Face backends (generative vs. discriminative).
- **Explainability:** Implementation of cross-modal similarity for visual grounding.
- **Software Engineering:** Clean separation of concerns (config, loader, pipeline, explainability).
- **Performance:** Hardware-aware profiling and lazy-loading implementations.
