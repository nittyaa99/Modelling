import streamlit as st
import joblib
import numpy as np
import re

from gensim.models import Word2Vec
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# LOAD MODEL


svm_model = joblib.load(
    'svm_skipgram.pkl'
)

label_encoder = joblib.load(
    'label_encoder.pkl'
)

w2v_model = Word2Vec.load(
    'word2vec_skipgram.model'
)


# PREPROCESSING


stopwords_indo = stopwords.words(
    'indonesian'
)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocessing(text):

    text = str(text)

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    text = text.lower()

    tokens = text.split()

    tokens = [
        word for word in tokens
        if word not in stopwords_indo
    ]

    tokens = [
        stemmer.stem(word)
        for word in tokens
    ]

    return tokens


# DOCUMENT VECTOR


def document_vector(tokens, model):

    vectors = []

    for word in tokens:

        if word in model.wv:
            vectors.append(model.wv[word])

    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


# PREDICTION


def predict_news(text):

    tokens = preprocessing(text)

    vector = document_vector(
        tokens,
        w2v_model
    )

    vector = vector.reshape(1, -1)

    pred = svm_model.predict(vector)

    label = label_encoder.inverse_transform(pred)

    return label[0]


# UI


st.title(
    "Klasifikasi Berita CNN Indonesia"
)

text = st.text_area(
    "Masukkan isi berita"
)

if st.button("Prediksi"):

    hasil = predict_news(text)

    st.success(
        f"Kategori: {hasil}"
    )