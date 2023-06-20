import streamlit as st
import pandas as pd
import string
import re
import scipy.sparse as sp
from PIL import Image

logo = Image.open('ikao.jpg')
st.image(logo, caption='')

st.title("Prediksi tweettt covid 19")
text = st.text_input("Masukkan teks")
button = st.button("Hasil Prediiksi")

if button:
    # Download rexource
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltk.download("punkt")
    nltk.download("stopwords")
    # membaca data
    df = pd.read_csv("https://github.com/davata1/pba/blob/main/covid.csv")

    # Mendefinisikan fungsi pra-pemrosesan
    def preprocess_text(text):
        # HTML Tag Removal
        text = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});").sub(
            "", str(text)
        )
        # Case folding
        text = text.lower()
        # Trim text
        text = text.strip()
        # Remove punctuations, karakter spesial, and spasi ganda
        text = re.compile("<.*?>").sub("", text)
        text = re.compile("[%s]" % re.escape(string.punctuation)).sub(" ", text)
        text = re.sub("\s+", " ", text)
        # Number removal
        text = re.sub(r"\[[0-9]*\]", " ", text)
        text = re.sub(r"[^\w\s]", "", str(text).lower().strip())
        text = re.sub(r"\d", " ", text)
        text = re.sub(r"\s+", " ", text)
        # Mengubah text 'nan' dengan whitespace agar nantinya dapat dihapus
        text = re.sub("nan", "", text)
        # Menghapus kata-kata yang tidak bermakna (stopwords)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words("Indonesian"))
        tokens = [token for token in tokens if token not in stop_words]
        # Menggabungkan kata-kata kembali menjadi teks yang telah dipreprocessed
        processed_text = " ".join(tokens)
        # Melakukan stemming pada teks menggunakan PySastrawi
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        stemmed_text = stemmer.stem(processed_text)
        return stemmed_text

    # Mengambil input teks dari pengguna
    # st.write("Hasil Preprocessing:")
    analisis = preprocess_text(text)
    # st.write(analisis)

    import pickle
    with open("modelKNNrill.pkl", "rb") as r:
        asknn = pickle.load(r)
    import pickle
    with open("tfidf.pkl", "rb") as f:
        vectoriz = pickle.load(f)

    tf = vectoriz.transform([analisis])
    predictions = asknn.predict(tf)
    for i in predictions:
        st.write("Text : ", analisis)
        st.write("Sentimenm analisis :", i)
    