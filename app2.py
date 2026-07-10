import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

from joblib import load
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =====================================
# CONFIG PAGE
# =====================================

st.set_page_config(
    page_title="NewsLens · Klasifikasi Berita",
    page_icon="🔍",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0B0F1A !important;
    color: #E8EDF5;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
[data-testid="stToolbar"] { display: none; }

/* ── Remove default padding ── */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Top bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 48px;
    background: rgba(255,255,255,0.03);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    backdrop-filter: blur(12px);
}

.topbar-logo {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.5px;
}

.topbar-logo span {
    color: #4F8EF7;
}

.topbar-badge {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #4F8EF7;
    background: rgba(79,142,247,0.12);
    border: 1px solid rgba(79,142,247,0.25);
    padding: 5px 12px;
    border-radius: 20px;
}

/* ── Hero section ── */
.hero {
    text-align: center;
    padding: 72px 48px 48px;
    max-width: 780px;
    margin: 0 auto;
}

.hero-eyebrow {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #4F8EF7;
    margin-bottom: 20px;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 52px;
    font-weight: 700;
    line-height: 1.1;
    color: #FFFFFF;
    margin-bottom: 18px;
    letter-spacing: -1.5px;
}

.hero-title em {
    color: #4F8EF7;
    font-style: normal;
}

.hero-sub {
    font-size: 16px;
    color: #8B96A8;
    line-height: 1.7;
    font-weight: 400;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(79,142,247,0.3), transparent);
    margin: 0 48px 48px;
}

/* ── Main layout ── */
.main-grid {
    display: grid;
    grid-template-columns: 1fr 320px;
    gap: 24px;
    padding: 0 48px 48px;
    max-width: 1200px;
    margin: 0 auto;
}

/* ── Card base ── */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 28px;
}

/* ── Textarea override ── */
.stTextArea label {
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    color: #8B96A8 !important;
    margin-bottom: 10px !important;
}

.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #E8EDF5 !important;
    font-size: 15px !important;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.7 !important;
    padding: 16px 18px !important;
    resize: none !important;
    transition: border-color 0.2s ease;
}

.stTextArea textarea:focus {
    border-color: rgba(79,142,247,0.5) !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,0.1) !important;
}

.stTextArea textarea::placeholder {
    color: #4A5568 !important;
}

/* ── Button override ── */
.stButton > button {
    width: 100% !important;
    background: #4F8EF7 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    height: 52px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.2px;
    transition: all 0.2s ease !important;
    margin-top: 14px !important;
}

.stButton > button:hover {
    background: #3B7BF0 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(79,142,247,0.35) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Sidebar info card ── */
.info-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}

.info-card-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #4F8EF7;
    margin-bottom: 16px;
}

.step-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 14px;
}

.step-number {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: rgba(79,142,247,0.15);
    border: 1px solid rgba(79,142,247,0.3);
    color: #4F8EF7;
    font-size: 11px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}

.step-text {
    font-size: 13.5px;
    color: #8B96A8;
    line-height: 1.55;
}

.tag-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 8px;
}

.tag {
    font-size: 11.5px;
    font-weight: 500;
    color: #8B96A8;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    padding: 4px 10px;
}

/* ── Result box ── */
.result-wrapper {
    margin-top: 24px;
    animation: fadeUp 0.4s ease;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-card {
    background: linear-gradient(135deg, rgba(79,142,247,0.12) 0%, rgba(79,142,247,0.04) 100%);
    border: 1px solid rgba(79,142,247,0.3);
    border-radius: 16px;
    padding: 32px;
    position: relative;
    overflow: hidden;
}

.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #4F8EF7, #7BB3FF);
    border-radius: 16px 16px 0 0;
}

.result-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4F8EF7;
    margin-bottom: 10px;
}

.result-category {
    font-family: 'Playfair Display', serif;
    font-size: 38px;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.1;
    letter-spacing: -1px;
}

.result-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid rgba(255,255,255,0.07);
}

.result-dot {
    width: 8px;
    height: 8px;
    background: #4DF0A0;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.result-meta-text {
    font-size: 13px;
    color: #8B96A8;
}

/* ── Warning / empty state ── */
.empty-state {
    background: rgba(255,200,80,0.06);
    border: 1px solid rgba(255,200,80,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 16px;
    font-size: 14px;
    color: #FFD166;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 32px 48px;
    border-top: 1px solid rgba(255,255,255,0.06);
    font-size: 13px;
    color: #4A5568;
    margin-top: 24px;
}

/* ── Hide streamlit elements ── */
.stAlert { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }

</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD MODEL
# =====================================

@st.cache_resource
def load_models():
    w2v_model = Word2Vec.load("word2vec_skipgram.model")
    svm_model = load("svm_skipgram.pkl")
    label_encoder = load("label_encoder.pkl")
    return w2v_model, svm_model, label_encoder

w2v_model, svm_model, label_encoder = load_models()

# =====================================
# PREPROCESSING
# =====================================

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

def cleaning(text):
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess(text):
    text = cleaning(text)
    text = text.lower()
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return tokens

# =====================================
# DOCUMENT VECTOR
# =====================================

def document_vector(model, doc):
    vectors = [model.wv[w] for w in doc if w in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# =====================================
# TOP BAR
# =====================================

st.markdown("""
<div class="topbar">
    <div class="topbar-logo">News<span>Lens</span></div>
    <div class="topbar-badge">Word2Vec · SVM</div>
</div>
""", unsafe_allow_html=True)

# =====================================
# HERO
# =====================================

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Klasifikasi Berita Otomatis</div>
    <div class="hero-title">Temukan <em>Kategori</em><br>Setiap Berita</div>
    <div class="hero-sub">
        Tempel teks berita apa pun, dan sistem akan menentukan kategorinya secara instan
        menggunakan model Word2Vec Skip-Gram yang dikombinasikan dengan Support Vector Machine.
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# =====================================
# MAIN LAYOUT — dua kolom Streamlit
# =====================================

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    berita = st.text_area(
        "Teks Berita",
        height=260,
        placeholder="Tempelkan isi berita di sini. Sistem mendukung artikel panjang maupun kutipan singkat..."
    )

    klasifikasi = st.button("🔍  Klasifikasikan Berita")

    if klasifikasi:
        if berita.strip() == "":
            st.markdown("""
            <div class="empty-state">
                ⚠️ &nbsp; Masukkan teks berita terlebih dahulu sebelum menjalankan klasifikasi.
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Memproses teks…"):
                tokens  = preprocess(berita)
                vector  = document_vector(w2v_model, tokens)
                pred    = svm_model.predict([vector])[0]
                kategori = label_encoder.inverse_transform([pred])[0]

            kata_count = len(berita.split())

            st.markdown(f"""
            <div class="result-wrapper">
                <div class="result-card">
                    <div class="result-label">Hasil Klasifikasi</div>
                    <div class="result-category">{kategori}</div>
                    <div class="result-meta">
                        <div class="result-dot"></div>
                        <div class="result-meta-text">
                            Klasifikasi selesai &nbsp;·&nbsp; {kata_count} kata diproses
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="info-card">
        <div class="info-card-title">Cara Penggunaan</div>
        <div class="step-item">
            <div class="step-number">1</div>
            <div class="step-text">Tempel atau ketik teks berita pada kolom di sebelah kiri.</div>
        </div>
        <div class="step-item">
            <div class="step-number">2</div>
            <div class="step-text">Klik tombol <strong style="color:#E8EDF5">Klasifikasikan Berita</strong> untuk memulai proses.</div>
        </div>
        <div class="step-item">
            <div class="step-number">3</div>
            <div class="step-text">Kategori berita akan muncul di bawah kolom teks secara otomatis.</div>
        </div>
    </div>

    <div class="info-card">
        <div class="info-card-title">Pipeline Pemrosesan</div>
        <div class="tag-row">
            <span class="tag">Pembersihan Teks</span>
            <span class="tag">Tokenisasi</span>
            <span class="tag">Stop-word Removal</span>
            <span class="tag">Stemming (Sastrawi)</span>
            <span class="tag">Word2Vec Skip-Gram</span>
            <span class="tag">Document Vector</span>
            <span class="tag">SVM Classifier</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================
# FOOTER
# =====================================

st.markdown("""
<div class="footer">
    NewsLens &nbsp;·&nbsp; Word2Vec Skip-Gram + SVM &nbsp;·&nbsp; Bahasa Indonesia
</div>
""", unsafe_allow_html=True)