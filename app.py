"""Streamlit UI for explainable visual question answering."""

from __future__ import annotations

from io import BytesIO
from typing import Optional

import streamlit as st
from PIL import Image, UnidentifiedImageError

from config import AppConfig, ExplainabilityConfig, ModelConfig, ModelProfile
from inference_pipeline import ConversationState, VQAInferencePipeline


def load_image_from_upload(uploaded_file: object) -> Image.Image:
    """Load an uploaded image and convert to RGB.

    Args:
        uploaded_file: The file object from st.file_uploader.

    Returns:
        A PIL RGB Image.

    Raises:
        ValueError: If the file is not a valid image.
    """
    try:
        data = uploaded_file.read()
        return Image.open(BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc


@st.cache_resource(show_spinner="Initializing model components...")
def get_pipeline(profile: ModelProfile) -> VQAInferencePipeline:
    """Create and cache the inference pipeline.

    Args:
        profile: Selected model profile.

    Returns:
        Initialized VQAInferencePipeline.
    """
    config = ModelConfig(model_profile=profile)
    return VQAInferencePipeline(
        model_config=config,
        explainability_config=ExplainabilityConfig(),
    )


def apply_custom_style() -> None:
    """Apply a restrained CSS theme for a research-oriented dashboard."""
    st.markdown(
        """
        <style>
        .main {
            background-color: #fcfcfc;
        }
        .stButton>button {
            width: 100%;
            border-radius: 4px;
        }
        .reportview-container .main .block-container {
            padding-top: 2rem;
        }
        h1, h2, h3 {
            color: #262730;
            font-weight: 400;
        }
        .stCaption {
            font-size: 0.9rem;
            color: #555;
        }
        div[data-testid="stExpander"] {
            border: 1px solid #f0f2f6;
            border-radius: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    """Initialize or reset the session state."""
    if "conversation" not in st.session_state:
        st.session_state.conversation = ConversationState()
    if "last_image_name" not in st.session_state:
        st.session_state.last_image_name = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "active_profile" not in st.session_state:
        st.session_state.active_profile = None


def reset_conversation_if_new_image(image_name: Optional[str]) -> None:
    """Reset the session state if a new image is detected."""
    if image_name and image_name != st.session_state.last_image_name:
        st.session_state.conversation = ConversationState()
        st.session_state.last_result = None
        st.session_state.last_image_name = image_name


def main() -> None:
    """Orchestrate the Streamlit application flow."""
    app_config = AppConfig()

    st.set_page_config(
        page_title=app_config.page_title,
        page_icon=app_config.page_icon,
        layout="wide",
    )

    apply_custom_style()
    initialize_state()

    st.title("Explainable VQA Dashboard")
    st.caption("A tool for visual reasoning and post-hoc semantic grounding.")

    # Sidebar: Model Configuration
    with st.sidebar:
        st.header("Configuration")
        
        profile_map = {
            "ViLT - fast CPU demo": "vilt",
            "BLIP-2 - heavier research mode": "blip2"
        }
        
        selected_label = st.selectbox(
            "Model Profile",
            options=list(profile_map.keys()),
            help="ViLT is optimized for speed and CPU usage. BLIP-2 offers generative multi-turn reasoning but requires more memory."
        )
        
        target_profile = profile_map[selected_label]
        
        # Detect profile change and reset state
        if st.session_state.active_profile and st.session_state.active_profile != target_profile:
            st.session_state.conversation = ConversationState()
            st.session_state.last_result = None
        
        st.session_state.active_profile = target_profile

        st.info(
            "ViLT: Fast classification, non-conversational.\n\n"
            "BLIP-2: Generative, supports context, resource intensive."
        )

        # Lazy loading status
        if "pipeline_loaded" not in st.session_state:
            st.session_state.pipeline_loaded = False
            st.warning("Model will load on the first question.")
        
        if st.session_state.pipeline_loaded:
            pipeline = get_pipeline(st.session_state.active_profile)
            models = pipeline.models
            st.divider()
            st.subheader("System Status")
            st.write(f"**VQA Profile:** `{models.model_profile}`")
            st.write(f"**VQA Model:** `{models.vqa_model.config._name_or_path.split('/')[-1]}`")
            st.write(f"**Device:** `{models.device}`")
            
            if models.model_profile == "blip2" and models.device.type == "cpu":
                st.warning("Warning: BLIP-2 on CPU can be extremely slow and may require large downloads.")

    # Main Layout
    left_col, right_col = st.columns([0.4, 0.6], gap="large")

    with left_col:
        st.subheader("Input")
        uploaded = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Maximum 10MB"
        )

        image: Optional[Image.Image] = None
        if uploaded is not None:
            reset_conversation_if_new_image(uploaded.name)
            try:
                image = load_image_from_upload(uploaded)
                st.image(image, width="stretch")
            except ValueError as exc:
                st.error(str(exc))

        question = st.text_input(
            "Question",
            placeholder="e.g., What is the object on the table?",
        )

        ask_btn = st.button(
            "Run Inference",
            type="primary",
            disabled=image is None or not question.strip(),
        )

        if st.button("Clear Conversation"):
            st.session_state.conversation = ConversationState()
            st.session_state.last_result = None
            st.rerun()

    with right_col:
        if ask_btn and image is not None:
            with st.spinner("Analyzing image and grounding evidence..."):
                try:
                    pipeline = get_pipeline(st.session_state.active_profile)
                    st.session_state.pipeline_loaded = True
                    
                    result = pipeline.answer_question(
                        image=image,
                        question=question,
                        state=st.session_state.conversation,
                    )
                    st.session_state.last_result = result
                except (RuntimeError, ValueError) as exc:
                    st.error(str(exc))

        result = st.session_state.last_result

        if result:
            with st.container(border=True):
                st.subheader("Answer")
                st.write(f"**{result.answer}**")
            
            st.subheader("Evidence")
            with st.container(border=True):
                st.image(
                    result.explanation.annotated_image,
                    caption=f"Visual Grounding: {result.explanation.grounding_phrase}",
                    width="stretch"
                )
                
                if not result.explanation.boxes:
                    st.info("No confident region crossed the threshold; inspect the heatmap qualitatively.")
                
                st.write(result.explanation.textual_explanation)

            st.subheader("Conversation History")
            if not st.session_state.conversation.turns:
                st.caption("No history available.")
            else:
                for idx, turn in enumerate(st.session_state.conversation.turns, start=1):
                    with st.expander(f"Turn {idx}: {turn.question}"):
                        st.write(turn.answer)


if __name__ == "__main__":
    main()
