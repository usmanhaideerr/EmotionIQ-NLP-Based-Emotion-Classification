import re
from html import escape
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Emotion AI | Text Emotion Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =====================================================
# PROJECT PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# =====================================================
# HELPER: RENDER HTML SAFELY
# Markdown turns lines indented by 4+ spaces into code
# blocks. This strips indentation and blank lines so
# the HTML is always rendered as HTML.
# =====================================================

def render_html(html: str):
    cleaned = re.sub(r"^[ \t]+", "", html, flags=re.MULTILINE)
    cleaned = re.sub(r"\n\s*\n", "\n", cleaned).strip()
    st.markdown(cleaned, unsafe_allow_html=True)


# =====================================================
# LOAD TRAINED MODEL, VECTORIZER AND LABELS
# =====================================================

@st.cache_resource
def load_models():
    model = joblib.load(MODEL_DIR / "svm_model2.pkl")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    labels = joblib.load(MODEL_DIR / "emotion_labels.pkl")
    return model, vectorizer, labels


try:
    model, vectorizer, emotion_labels = load_models()

except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()

except Exception as e:
    st.error(f"Unable to load trained models: {e}")
    st.stop()


# =====================================================
# EMOTION STYLES (emoji + accent color)
# =====================================================

EMOTION_STYLE = {
    "joy": ("😄", "#fbbf24"),
    "happy": ("😄", "#fbbf24"),
    "happiness": ("😄", "#fbbf24"),
    "sadness": ("😢", "#60a5fa"),
    "sad": ("😢", "#60a5fa"),
    "anger": ("😡", "#f87171"),
    "angry": ("😡", "#f87171"),
    "fear": ("😨", "#a78bfa"),
    "love": ("❤️", "#f472b6"),
    "surprise": ("😮", "#34d399"),
    "disgust": ("🤢", "#a3e635"),
    "neutral": ("😐", "#94a3b8"),
}

DEFAULT_STYLE = ("🧠", "#2dd4bf")


def get_style(emotion_name: str):
    return EMOTION_STYLE.get(emotion_name.strip().lower(), DEFAULT_STYLE)


def label_name(value) -> str:
    """Convert a numeric class into its emotion name."""
    if hasattr(value, "item"):
        value = value.item()
    return str(
        emotion_labels.get(
            value,
            emotion_labels.get(str(value), value),
        )
    )


def get_confidence(text_vector, top_n=3):
    """Return the top emotions with confidence scores (in %)."""
    try:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(text_vector)[0]
        else:
            raw = model.decision_function(text_vector)
            raw = np.atleast_2d(raw)[0]
            if raw.shape[0] == 1:
                raw = np.array([-raw[0], raw[0]])
            exp = np.exp(raw - raw.max())
            scores = exp / exp.sum()

        order = np.argsort(scores)[::-1][:top_n]
        return [
            (label_name(model.classes_[i]), float(scores[i]) * 100)
            for i in order
        ]
    except Exception:
        return []


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
}

/* Main Background */

.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(20, 184, 166, 0.14), transparent 40%),
        radial-gradient(circle at 85% 90%, rgba(99, 102, 241, 0.14), transparent 40%),
        linear-gradient(135deg, #0a1020, #0f1a2e);
    color: #f1f5f9;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Hero */

.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(120deg, #172b46, #134e4a);
    border: 1px solid #2b5d73;
    border-radius: 24px;
    padding: 42px;
    margin-bottom: 32px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35);
}

.hero::after {
    content: "";
    position: absolute;
    top: -80px;
    right: -80px;
    width: 260px;
    height: 260px;
    background: radial-gradient(circle, rgba(45, 212, 191, 0.28), transparent 70%);
}

.hero-tag {
    display: inline-block;
    color: #75e5d2;
    background: rgba(20, 59, 67, 0.9);
    border: 1px solid #286b70;
    padding: 7px 14px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin-bottom: 16px;
}

.hero h1 {
    font-size: 46px;
    font-weight: 800;
    margin: 0 0 12px 0;
    padding: 0;
    background: linear-gradient(90deg, #ffffff, #99f6e4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #c0d4e5;
    font-size: 16px;
    line-height: 1.7;
    margin: 0 0 6px 0;
    max-width: 640px;
}

.hero-stats {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 22px;
}

.hero-chip {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: #e2e8f0;
    padding: 7px 14px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
}

/* Section Headings */

.section-heading {
    color: #ffffff;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 14px;
}

/* Cards */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(18, 30, 48, 0.85);
    border: 1px solid #293b52;
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
}

/* Text Area */

div[data-testid="stTextArea"] textarea {
    background: #0b1728 !important;
    color: #ffffff !important;
    border: 1px solid #34465e !important;
    border-radius: 12px !important;
    font-size: 15px;
}

div[data-testid="stTextArea"] textarea:focus {
    border: 1px solid #14b8a6 !important;
    box-shadow: 0 0 0 2px rgba(20, 184, 166, 0.35) !important;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(135deg, #14b8a6, #0f766e);
    color: white;
    border: none;
    border-radius: 11px;
    min-height: 45px;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #0f9488, #115e59);
    color: white;
    border: none;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(20, 184, 166, 0.35);
}

.stButton > button:active {
    transform: translateY(0);
}

/* Example Buttons (secondary look) */

div[data-testid="stHorizontalBlock"] .stButton > button {
    font-size: 13px;
}

/* Character counter */

.counter {
    text-align: right;
    color: #6b819e;
    font-size: 12px;
    margin-top: -6px;
    margin-bottom: 10px;
}

/* Result Card */

.result-card {
    border-radius: 20px;
    padding: 34px 24px;
    text-align: center;
    margin-top: 28px;
    background: linear-gradient(120deg, #123b3c, #174c4b);
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.35);
}

.result-emoji {
    font-size: 56px;
    line-height: 1;
    margin-bottom: 10px;
}

.result-label {
    color: #a7f3d0;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
}

.result-emotion {
    color: #ffffff;
    font-size: 42px;
    font-weight: 800;
    margin-top: 8px;
    text-transform: capitalize;
}

/* Confidence bars */

.conf-wrap {
    margin-top: 22px;
    background: rgba(18, 30, 48, 0.85);
    border: 1px solid #293b52;
    border-radius: 18px;
    padding: 22px 26px;
}

.conf-title {
    color: #ffffff;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 14px;
}

.conf-row {
    margin-bottom: 14px;
}

.conf-head {
    display: flex;
    justify-content: space-between;
    color: #dbe7f3;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 6px;
    text-transform: capitalize;
}

.conf-track {
    width: 100%;
    height: 10px;
    background: #0b1728;
    border-radius: 20px;
    overflow: hidden;
}

.conf-fill {
    height: 100%;
    border-radius: 20px;
}

/* About Information */

.info-item {
    padding: 10px 0;
    border-bottom: 1px solid #22344b;
}

.info-item:last-child {
    border-bottom: none;
}

.info-label {
    color: #7f97b3;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    margin-bottom: 3px;
}

.info-value {
    color: #f1f5f9;
    font-size: 15px;
    font-weight: 600;
}

/* Footer */

.footer {
    text-align: center;
    color: #7f93ad;
    padding-top: 10px;
    font-size: 13px;
}

.footer p {
    margin: 4px 0;
}

.footer-line {
    height: 1px;
    background: linear-gradient(90deg, transparent, #293b52, transparent);
    margin: 36px 0 18px 0;
}

/* Hide Streamlit Branding */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# HEADER
# =====================================================

render_html("""
<div class="hero">
    <div class="hero-tag">MACHINE LEARNING PROJECT</div>
    <h1>🧠 Emotion AI</h1>
    <p>Understand the emotions hidden behind human language with intelligent text analysis.</p>
    <p>Enter a sentence and let our machine learning model detect its emotional category.</p>
    <div class="hero-stats">
        <div class="hero-chip">⚡ Instant results</div>
        <div class="hero-chip">📊 Confidence scores</div>
        <div class="hero-chip">🤖 Linear SVM + TF-IDF</div>
    </div>
</div>
""")


# =====================================================
# EXAMPLE SENTENCES
# =====================================================

examples = [
    "I am so happy because I achieved my goals!",
    "I feel very sad and lonely today.",
    "I am extremely angry about this situation.",
    "I am scared about what might happen next.",
]


# =====================================================
# CALLBACKS
# =====================================================

def select_example(text):
    st.session_state["user_text"] = text
    st.session_state.pop("prediction_result", None)


def clear_text():
    st.session_state["user_text"] = ""
    st.session_state.pop("prediction_result", None)


# =====================================================
# MAIN LAYOUT
# =====================================================

left, right = st.columns([1.6, 1], gap="large")


# =====================================================
# TEXT INPUT SECTION
# =====================================================

with left:

    render_html('<div class="section-heading">✍️ Analyze Your Text</div>')

    with st.container(border=True):

        st.write("Enter your sentence below")

        st.caption("Try an example:")

        example_cols = st.columns(4)

        for i, example in enumerate(examples):
            with example_cols[i]:
                st.button(
                    f"Example {i + 1}",
                    key=f"example_{i}",
                    use_container_width=True,
                    on_click=select_example,
                    args=(example,),
                )

        user_text = st.text_area(
            "Your text",
            key="user_text",
            placeholder="Example: I am feeling very happy today!",
            height=160,
            max_chars=500,
            label_visibility="collapsed",
        )

        render_html(f'<div class="counter">{len(user_text)} / 500 characters</div>')

        col1, col2 = st.columns([2, 1])

        with col1:
            analyze = st.button("✨ Detect Emotion", use_container_width=True)

        with col2:
            st.button(
                "Clear Text",
                use_container_width=True,
                on_click=clear_text,
            )

    st.caption("Tip: Write a complete sentence for better prediction results.")


# =====================================================
# ABOUT THIS APP
# =====================================================

with right:

    render_html('<div class="section-heading">🔍 About This App</div>')

    with st.container(border=True):

        render_html("""
        <div class="info-item">
            <div class="info-label">CLASSIFICATION MODEL</div>
            <div class="info-value">Linear Support Vector Machine</div>
        </div>
        <div class="info-item">
            <div class="info-label">FEATURE EXTRACTION</div>
            <div class="info-value">TF-IDF Vectorization</div>
        </div>
        <div class="info-item">
            <div class="info-label">MACHINE LEARNING TASK</div>
            <div class="info-value">Emotion Classification</div>
        </div>
        <div class="info-item">
            <div class="info-label">INPUT TYPE</div>
            <div class="info-value">English Text</div>
        </div>
        <div class="info-item">
            <div class="info-label">MODEL OUTPUT</div>
            <div class="info-value">Predicted Emotion Category</div>
        </div>
        """)


# =====================================================
# EMOTION PREDICTION
# =====================================================

if analyze:

    if not user_text.strip():

        st.warning("Please enter some text before detecting emotion.")
        st.session_state.pop("prediction_result", None)

    else:

        try:

            with st.spinner("Analyzing your text..."):

                text_vector = vectorizer.transform([user_text])

                prediction = model.predict(text_vector)[0]

                emotion = label_name(prediction)

                st.session_state["prediction_result"] = {
                    "emotion": emotion,
                    "scores": get_confidence(text_vector),
                }

        except Exception as e:

            st.session_state.pop("prediction_result", None)
            st.error(f"Prediction failed: {e}")


# =====================================================
# DISPLAY PREDICTION RESULT
# =====================================================

if "prediction_result" in st.session_state:

    result = st.session_state["prediction_result"]

    emotion_name = result["emotion"]
    emoji, color = get_style(emotion_name)

    render_html(f"""
    <div class="result-card" style="border: 1px solid {color}; box-shadow: 0 0 40px {color}33;">
        <div class="result-emoji">{emoji}</div>
        <div class="result-label">DETECTED EMOTION</div>
        <div class="result-emotion" style="color: {color};">{escape(emotion_name)}</div>
    </div>
    """)

    if result["scores"]:

        rows = ""

        for name, pct in result["scores"]:
            _, bar_color = get_style(name)
            rows += (
                '<div class="conf-row">'
                f'<div class="conf-head"><span>{escape(name)}</span><span>{pct:.1f}%</span></div>'
                '<div class="conf-track">'
                f'<div class="conf-fill" style="width:{pct:.1f}%; background:{bar_color};"></div>'
                '</div>'
                '</div>'
            )

        render_html(f"""
        <div class="conf-wrap">
            <div class="conf-title">📊 Confidence Breakdown</div>
            {rows}
        </div>
        """)

    st.success("Text analysis completed successfully!")


# =====================================================
# FOOTER
# =====================================================

render_html("""
<div class="footer-line"></div>
<div class="footer">
    <p>Emotion AI | Powered by Machine Learning</p>
    <p>Linear SVM · TF-IDF · Natural Language Processing</p>
</div>
""")