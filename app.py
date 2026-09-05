import os
import re
import time
import joblib
import nltk
import torch
import torch.nn as nn
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources silently
nltk.download('stopwords', quiet=True)

# Workaround for duplicate OpenMP runtime crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==========================================
# 1. PAGE CONFIGURATION & AMAZON THEME STYLING
# ==========================================
st.set_page_config(
    page_title="Amazon Sentiment AI | Deep Learning Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS matching Amazon Brand Identity
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #0f141d;
        color: #eaeded;
    }

    /* Main Container Spacing */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1150px;
    }

    /* Sidebar Amazon Theme */
    [data-testid="stSidebar"] {
        background-color: #131921 !important;
        border-right: 1px solid #232f3e;
    }

    /* Hero Banner Header (Amazon Dark Navy Gradient) */
    .hero-container {
        background: linear-gradient(135deg, #131921 0%, #232f3e 100%);
        border: 1px solid #37475a;
        border-radius: 16px;
        padding: 2.2rem 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        margin-bottom: 2rem;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ff9900 0%, #febd69 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        color: #aab7c4;
        font-size: 1.02rem;
        font-weight: 300;
        margin-bottom: 0;
    }

    /* Result Metric Cards */
    .result-card-pos {
        background: linear-gradient(135deg, rgba(6, 125, 98, 0.25) 0%, rgba(19, 25, 33, 0.95) 100%);
        border: 1.5px solid #067d62;
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        color: #2ed573;
        box-shadow: 0 4px 20px rgba(6, 125, 98, 0.2);
    }

    .result-card-neg {
        background: linear-gradient(135deg, rgba(186, 24, 27, 0.25) 0%, rgba(19, 25, 33, 0.95) 100%);
        border: 1.5px solid #ba181b;
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        color: #ff6b6b;
        box-shadow: 0 4px 20px rgba(186, 24, 27, 0.2);
    }

    .result-card-neu {
        background: linear-gradient(135deg, rgba(217, 119, 6, 0.25) 0%, rgba(19, 25, 33, 0.95) 100%);
        border: 1.5px solid #d97706;
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        color: #f1c40f;
        box-shadow: 0 4px 20px rgba(217, 119, 6, 0.2);
    }

    /* Input Controls */
    .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid #37475a;
        background-color: #131921;
        font-size: 0.98rem;
        color: #eaeded;
        transition: all 0.3s ease;
    }

    .stTextArea textarea:focus {
        border-color: #ff9900;
        box-shadow: 0 0 10px rgba(255, 153, 0, 0.3);
    }

    /* Primary Action Button (Amazon Orange Call-To-Action) */
    .stButton>button {
        width: 100%;
        background: linear-gradient(180deg, #f8e3ad 0%, #ffa41c 100%);
        color: #111111;
        font-weight: 700;
        font-size: 1rem;
        border: 1px solid #a88734;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .stButton>button:hover {
        background: linear-gradient(180deg, #f7dfa5 0%, #f08804 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(255, 153, 0, 0.4);
        color: #000;
    }

    /* Progress Bar Accent */
    .stProgress > div > div > div > div {
        background-color: #ff9900;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. MODEL ARCHITECTURE & PREPROCESSING LOGIC
# ==========================================
class AmazonSentimentRNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def clean_text(text):
    """Replicates exact notebook text normalization pipeline."""
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    tokens = text.split()
    cleaned_tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)


@st.cache_resource
def load_artifacts():
    """Loads TF-IDF vectorizer and trained PyTorch weights into memory."""
    vectorizer = joblib.load('tfidf_vectorizer_amazon.joblib')
    input_size = len(vectorizer.get_feature_names_out())
    model = AmazonSentimentRNN(input_size=input_size)
    model.load_state_dict(torch.load('rnn_amazon_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return vectorizer, model


def predict_sentiment(raw_review, vectorizer, model):
    """Runs full pipeline: raw text -> clean -> tfidf -> tensor -> model forward pass."""
    cleaned = clean_text(raw_review)
    features = vectorizer.transform([cleaned]).toarray()
    tensor_input = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
    
    with torch.no_grad():
        logits = model(tensor_input)
        prob = torch.sigmoid(logits.squeeze()).item()
        
    return prob, cleaned


# ==========================================
# 3. SIDEBAR CONTROLS & BRAND BADGE
# ==========================================
with st.sidebar:
    # Amazon Branded Badge
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(255, 153, 0, 0.15) 0%, rgba(35, 47, 62, 0.9) 100%);
        border: 1px solid rgba(255, 153, 0, 0.4);
        padding: 12px 16px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    ">
        <img src="https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg" width="46" style="filter: brightness(0) invert(1) drop-shadow(0 2px 4px rgba(255,153,0,0.5));">
        <div>
            <div style="font-weight: 700; font-size: 1.05rem; color: #ffffff; letter-spacing: -0.3px;">Amazon AI</div>
            <div style="font-size: 0.7rem; color: #ff9900; font-weight: 700; letter-spacing: 0.5px;">SENTIMENT STUDIO</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Model Specs Section
    st.markdown("### ⚙️ Engine Specs")
    st.markdown("""
    - **Architecture:** PyTorch LSTM (`nn.LSTM`)
    - **Feature Extractor:** TF-IDF Vectorizer
    - **Test Accuracy:** `79.0%`
    - **Class 0 (Neg) F1:** `0.83`
    - **Class 1 (Pos) F1:** `0.74`
    """)

    st.markdown("---")
    
    # Preset Samples Section
    st.markdown("### 🧪 Quick Presets")
    st.caption("Click to auto-populate test reviews:")

    if st.button("🌟 5-Star Review"):
        st.session_state.review_text = "Absolutely amazing product! Excellent build quality, lightning fast shipping, and top notch performance."

    if st.button("⚠️ 1-Star Review"):
        st.session_state.review_text = "Extremely terrible experience. Arrived broken, materials feel cheap, and customer support refused a refund."

    if st.button("🤔 Mixed Review"):
        st.session_state.review_text = "The product looks nice and works reasonably well, but the setup process was confusing and took hours."

    st.markdown("---")
    st.caption("Developed with PyTorch & Streamlit")


# ==========================================
# 4. MAIN USER INTERFACE
# ==========================================

# Hero Header Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-title">📦 Amazon Review Sentiment Intelligence</div>
    <div class="hero-subtitle">Evaluate customer review sentiment in real-time using PyTorch Recurrent Neural Networks</div>
</div>
""", unsafe_allow_html=True)

# Main Two-Column Layout
col_input, col_output = st.columns([1.2, 1], gap="large")

with col_input:
    st.subheader("📝 Input Review")
    
    default_text = st.session_state.get("review_text", "")
    review_input = st.text_area(
        label="Enter or paste customer feedback below:",
        value=default_text,
        height=210,
        placeholder="e.g. The build quality feels cheap and plastic, definitely not worth the price tag..."
    )

    analyze_btn = st.button("Analyze Sentiment ✨")

with col_output:
    st.subheader("📊 Model Inference")

    if analyze_btn:
        if not review_input.strip():
            st.warning("⚠️ Please provide text before running sentiment analysis.")
        else:
            with st.spinner("Running text normalization & neural inference..."):
                time.sleep(0.3)
                vectorizer, model = load_artifacts()
                probability, cleaned_text = predict_sentiment(review_input, vectorizer, model)

            is_pos = probability >= 0.5
            confidence = probability if is_pos else (1.0 - probability)

            # Sentiment Header Card with Neutral/Mixed Range (0.42 to 0.58)
            if 0.42 <= probability <= 0.58:
                st.markdown(f"""
                <div class="result-card-neu">
                    <h2 style="margin:0;">🤔 MIXED / NEUTRAL REVIEW</h2>
                    <p style="margin:6px 0 0 0; font-size: 0.95rem; font-weight: 600;">
                        Balanced Signals Identified ({probability * 100:.1f}% Positivity)
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif probability > 0.58:
                st.markdown(f"""
                <div class="result-card-pos">
                    <h2 style="margin:0;">😃 POSITIVE REVIEW</h2>
                    <p style="margin:6px 0 0 0; font-size: 0.95rem; font-weight: 600;">
                        Confidence Score: {confidence * 100:.2f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-neg">
                    <h2 style="margin:0;">😞 NEGATIVE REVIEW</h2>
                    <p style="margin:6px 0 0 0; font-size: 0.95rem; font-weight: 600;">
                        Confidence Score: {confidence * 100:.2f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            
            # Metric Columns
            col_pos_val, col_neg_val = st.columns(2)
            col_pos_val.metric("Positive Prob", f"{probability * 100:.1f}%")
            col_neg_val.metric("Negative Prob", f"{(1 - probability) * 100:.1f}%")

            # Probability Gauge
            st.markdown("**Positivity Probability Distribution**")
            st.progress(probability)

            # Preprocessed Text Accordion
            with st.expander("🔍 View Preprocessed Text Output"):
                st.write("**Original Text:**")
                st.caption(review_input)
                st.write("**Cleaned & Stemmed Tokens:**")
                st.code(cleaned_text if cleaned_text else "[No valid tokens remaining after stopword removal]", language="text")

    else:
        # Default Welcome State
        st.info("👈 Enter a customer review on the left or click a **Quick Preset** in the sidebar to view prediction insights.")