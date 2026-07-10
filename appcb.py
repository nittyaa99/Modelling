import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

from joblib import load
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# CONFIG PAGE

st.set_page_config(
    page_title="NewsLens · Klasifikasi Berita",
    page_icon="🔍",
    layout="wide"
)

# CUSTOM CSS — Light Theme

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    background: #F0F4FF !important;
    color: #1A2340;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
[data-testid="stToolbar"] { display: none; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Top bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 48px;
    background: #FFFFFF;
    border-bottom: 1px solid #E2E8F0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

.topbar-logo {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: #1A2340;
    letter-spacing: -0.5px;
}

.topbar-logo span { color: #2563EB; }

.topbar-badge {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #2563EB;
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    padding: 5px 14px;
    border-radius: 20px;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 64px 48px 40px;
    max-width: 760px;
    margin: 0 auto;
}

.hero-eyebrow {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #2563EB;
    margin-bottom: 18px;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 50px;
    font-weight: 700;
    line-height: 1.1;
    color: #0F172A;
    margin-bottom: 18px;
    letter-spacing: -1.5px;
}

.hero-title em {
    color: #2563EB;
    font-style: normal;
    position: relative;
}

.hero-sub {
    font-size: 16px;
    color: #64748B;
    line-height: 1.75;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #BFDBFE, transparent);
    margin: 0 48px 44px;
}

/* ── Textarea ── */
.stTextArea label {
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    color: #64748B !important;
    margin-bottom: 8px !important;
}

.stTextArea textarea {
    background: #FFFFFF !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 14px !important;
    color: #1A2340 !important;
    font-size: 15px !important;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.75 !important;
    padding: 16px 18px !important;
    resize: none !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}

.stTextArea textarea:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}

.stTextArea textarea::placeholder { color: #CBD5E1 !important; }

/* ── Button ── */
.stButton > button {
    width: 100% !important;
    background: #2563EB !important;
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
    box-shadow: 0 4px 14px rgba(37,99,235,0.3) !important;
}

.stButton > button:hover {
    background: #1D4ED8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 22px rgba(37,99,235,0.4) !important;
}

.stButton > button:active { transform: translateY(0) !important; }

/* ── Info cards (right column) ── */
.info-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.info-card-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #2563EB;
    margin-bottom: 18px;
}

.step-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 14px;
}

.step-number {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: #EFF6FF;
    border: 1.5px solid #BFDBFE;
    color: #2563EB;
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
    color: #475569;
    line-height: 1.6;
}

.tag-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.tag {
    font-size: 11.5px;
    font-weight: 500;
    color: #475569;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 4px 10px;
}

/* ── Result card ── */
.result-wrapper {
    margin-top: 20px;
    animation: fadeUp 0.35s ease;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-card {
    background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
    border: 1.5px solid #93C5FD;
    border-radius: 16px;
    padding: 30px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(37,99,235,0.1);
}

.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #2563EB, #60A5FA);
    border-radius: 16px 16px 0 0;
}

.result-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #2563EB;
    margin-bottom: 10px;
}

.result-category {
    font-family: 'Playfair Display', serif;
    font-size: 40px;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.1;
    letter-spacing: -1px;
}

.result-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 18px;
    padding-top: 16px;
    border-top: 1px solid rgba(37,99,235,0.15);
}

.result-dot {
    width: 8px;
    height: 8px;
    background: #16A34A;
    border-radius: 50%;
    animation: pulse 2s infinite;
    flex-shrink: 0;
}

@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(22,163,74,0.4); }
    50%       { opacity: 0.7; box-shadow: 0 0 0 4px rgba(22,163,74,0); }
}

.result-meta-text {
    font-size: 13px;
    color: #64748B;
}

/* ── Empty state ── */
.empty-state {
    background: #FFFBEB;
    border: 1px solid #FCD34D;
    border-radius: 12px;
    padding: 14px 18px;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 14px;
    font-size: 14px;
    color: #92400E;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 28px 48px;
    border-top: 1px solid #E2E8F0;
    font-size: 13px;
    color: #94A3B8;
    margin-top: 32px;
    background: #FFFFFF;
}

/* ── Hide default Streamlit chrome ── */
.stAlert { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }

</style>
""", unsafe_allow_html=True)


# LOAD MODEL
from pathlib import Path

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_models():
    w2v_model = Word2Vec.load(BASE_DIR / "word2vec_skipgram.model")
    svm_model = load(BASE_DIR / "svm_skipgram.pkl")
    label_encoder = load(BASE_DIR / "label_encoder.pkl")
    return w2v_model, svm_model, label_encoder

w2v_model, svm_model, label_encoder = load_models()


# ==========================
# PREPROCESSING
# ==========================

import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('stopwords', quiet=True)

# Inisialisasi
stop_words = set(stopwords.words('indonesian'))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# 1. Cleaning
def cleaning(teks):
    teks = re.sub(r'http\S+|www\.\S+', ' ', teks)
    teks = re.sub(r'[^a-zA-Z\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

# 2. Case Folding
def case_folding(teks):
    return teks.lower()

# 3. Tokenization
def tokenization(teks):
    return teks.split()

# 4. Stopword Removal
def stopword_removal(tokens):
    return [word for word in tokens if word not in stop_words]

# 5. Stemming
def stemming(tokens):
    return [stemmer.stem(word) for word in tokens]

# Fungsi Preprocessing
def preprocess(text):
    text = cleaning(text)
    text = case_folding(text)
    tokens = tokenization(text)
    tokens = stopword_removal(tokens)
    tokens = stemming(tokens)
    return tokens

# DOCUMENT VECTOR

def document_vector(model, doc):
    vectors = [model.wv[w] for w in doc if w in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


# TOP BAR


st.markdown("""
<div class="topbar">
    <div class="topbar-logo">News<span>Lens</span></div>
    <div class="topbar-badge">Word2Vec · SVM</div>
</div>
""", unsafe_allow_html=True)


# HERO


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


# MAIN LAYOUT


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
                ⚠️&nbsp; Masukkan teks berita terlebih dahulu sebelum menjalankan klasifikasi.
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Memproses teks…"):
                tokens   = preprocess(berita)
                vector   = document_vector(w2v_model, tokens)
                pred     = svm_model.predict([vector])[0]
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
            <div class="step-text">Klik tombol <strong style="color:#1A2340">Klasifikasikan Berita</strong> untuk memulai proses.</div>
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


# FOOTER


st.markdown("""
<div class="footer">
    NewsLens &nbsp;·&nbsp; Word2Vec Skip-Gram + SVM &nbsp;·&nbsp; Bahasa Indonesia
</div>
""", unsafe_allow_html=True)
