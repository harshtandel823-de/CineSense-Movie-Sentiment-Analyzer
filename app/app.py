
import os
import sys
import joblib
import streamlit as st


# PATHS AND MODEL LOADING

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.append(BASE_DIR)

from src.preprocessing import preprocess_text

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "sentiment_model.pkl"
)
VECTORIZER_PATH = os.path.join(
    BASE_DIR, "models", "tfidf_vectorizer.pkl"
)

st.set_page_config(
    page_title="CineSense | Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_models()


# CUSTOM STYLING

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #101426 0%, #171d35 100%);
        color: #f3f4f6;
    }

    [data-testid="stSidebar"] {
        background-color: #0b1020;
    }

    .hero {
        padding: 28px 30px;
        border: 1px solid #303958;
        border-radius: 18px;
        background: linear-gradient(120deg, #202a4a, #171d35);
        margin-bottom: 24px;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 2.5rem;
        margin-bottom: 8px;
    }

    .hero p {
        color: #c5cbe0;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    .result-card {
        padding: 24px;
        border-radius: 16px;
        margin-top: 18px;
        border: 1px solid #39435f;
        background: #1c2440;
    }

    .positive {
        border-left: 6px solid #22c55e;
    }

    .negative {
        border-left: 6px solid #f87171;
    }

    .muted {
        color: #b5bfd8;
        font-size: 0.9rem;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 46px;
    }

    div[data-testid="stMetric"] {
        background: #1c2440;
        padding: 16px;
        border: 1px solid #303958;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# SIDEBAR

with st.sidebar:
    st.title("🎬 CineSense")
    st.caption("Movie Review Intelligence")

    st.divider()

    st.subheader("About this project")
    st.write(
        "CineSense uses Natural Language Processing "
        "and Machine Learning to classify movie reviews "
        "as positive or negative."
    )

    st.subheader("Model information")
    st.write("**Classifier:** Linear Support Vector Machine")
    st.write("**Text features:** TF-IDF")
    st.write("**N-grams:** Unigrams and bigrams")
    st.write("**Dataset:** NLTK Movie Reviews")

    st.divider()
    st.caption("Minor Project | NLP + Machine Learning")


# HEADER

st.markdown("""
<div class="hero">
    <h1>🎥 Movie Sentiment Analyzer</h1>
    <p>
        Discover the sentiment behind a movie review.
        Enter your text and let the trained machine learning
        model classify it.
    </p>
</div>
""", unsafe_allow_html=True)


# TOP METRICS

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Task", "Sentiment Analysis")

with col2:
    st.metric("Classes", "2")

with col3:
    st.metric("Model", "Linear SVM")

st.divider()


# REVIEW INPUT

st.subheader("📝 Analyze a movie review")
st.write("Enter your own review or try one of the examples.")

sample_reviews = {
    "Choose an example...": "",
    "Positive review": (
        "The movie was fantastic. The acting was brilliant, "
        "the story was engaging, and I loved every minute."
    ),
    "Negative review": (
        "The movie was boring and disappointing. "
        "The acting was poor and the story was predictable."
    ),
}

selected_sample = st.selectbox(
    "Try a sample review",
    list(sample_reviews.keys()),
)

default_review = sample_reviews[selected_sample]

review = st.text_area(
    "Your review",
    value=default_review,
    height=180,
    placeholder="Example: The movie was fantastic!",
)

word_count = len(review.split())
character_count = len(review)

metric1, metric2 = st.columns(2)
metric1.caption(f"📝 Words: {word_count}")
metric2.caption(f"🔤 Characters: {character_count}")

analyze = st.button(
    "🔍 Analyze Sentiment",
    type="primary",
    use_container_width=True,
)


# PREDICTION

if analyze:
    if not review.strip():
        st.warning("Please enter a movie review before analyzing.")

    else:
        with st.spinner("Analyzing your review..."):
            cleaned_review = preprocess_text(review)
            review_vector = vectorizer.transform([cleaned_review])
            prediction = model.predict(review_vector)[0]

        st.divider()
        st.subheader("📊 Analysis Result")

        if prediction == "pos":
            st.markdown("""
            <div class="result-card positive">
                <h2>😊 Positive Sentiment</h2>
                <p>
                    The model classified this review as positive.
                    The review expresses a favorable sentiment
                    according to the trained classifier.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="result-card negative">
                <h2>😞 Negative Sentiment</h2>
                <p>
                    The model classified this review as negative.
                    The review expresses an unfavorable sentiment
                    according to the trained classifier.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.caption(
            "This prediction is a machine learning classification, "
            "not a guarantee of the review's true sentiment."
        )


# FOOTER

st.divider()

st.markdown(
    "<p class='muted'>CineSense | Movie Review Sentiment Analysis"
    " | Built using Python, Streamlit, NLP and Machine Learning</p>",
    unsafe_allow_html=True,
)
