import streamlit as st
import pandas as pd
import string
import re
from PIL import Image

logo = Image.open('https://img.freepik.com/free-vector/person-fighting-virus-illustrated_52683-35833.jpg?size=626&ext=jpg&ga=GA1.2.914826625.1687207747&semt=ais')
st.image(logo, caption='')

st.title("Prediksi tweettt covid 19")
text = st.text_input("Masukkan teks")
button = st.button("Hasil Prediiksi")

if button:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    # Menginisialisasi Streamlit
    # st.title("Preprocessing pada Teks"

    # Mengaktifkan resource NLTK yang diperlukan
    nltk.download("punkt")
    nltk.download("stopwords")
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    # Membaca kamus dari file Excel
    df = pd.read_csv("https://github.com/davata1/pba/blob/main/covid.csv")
    # Mengubah kamus menjadi dictionary

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
        stop_words = set(stopwords.words("Indonesian"))
        tokens = [token for token in tokens if token not in stop_words]

        # Menggabungkan kata-kata kembali menjadi teks yang telah dipreprocessed
        processed_text = " ".join(tokens)

        # Melakukan stemming pada teks menggunakan PySastrawi
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
        st.write("Sentimenm :", i)
    # Menampilkan hasil prediksi
    # sentiment = asknn.predict(cosim)
    # st.write("Sentimen:", sentiment)
