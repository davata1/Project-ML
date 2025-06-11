import pickle
import pandas as pd
import numpy as np
import re
import string
import nltk
import streamlit as st

# Download NLTK data seperti di Google Colab
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from skmultilearn.problem_transform import LabelPowerset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss, f1_score
import joblib

with st.container():
    st.title('Aplikasi Analisis Sentimen Berbasis Aspek :red[Ulasan Rumah Makan Bebek Sinjay]')
    input = st.text_area("**Masukkan Ulasan**")
    submit = st.button("Proses", type="primary")

    if submit:
        df_mentah = pd.DataFrame({'Ulasan': [input]})

        data = pd.read_excel('https://raw.githubusercontent.com/davata1/Project-ML/main/sinjaymadura.xlsx')


        def case_fold(text):
            return text.lower()

        def remove_punctuation(text):
            data = re.sub('@[^\s]+', ' ', text)
            data = re.sub(r'http\S*', ' ', data)
            data = data.translate(str.maketrans(' ', ' ', string.punctuation))
            data = re.sub('[^a-zA-Z]', ' ', data)
            data = re.sub("\n", " ", data)
            data = re.sub(r"\b[a-zA-z]\b", " ", data)
            return data

        def tokenize(text):
            return nltk.word_tokenize(text)

        kamus_normalisasi = pd.read_csv('https://raw.githubusercontent.com/davata1/Project-ML/main/colloquial-indonesian-lexicon.csv')
        kamus_normalisasi = kamus_normalisasi.drop(columns=['In-dictionary', 'context', 'category1', 'category2', 'category3'])

        def normalization(token):
            ulasan_normalisasi = []
            for kata in token:
                if kata in kamus_normalisasi['slang'].values:
                    formal = kamus_normalisasi.loc[kamus_normalisasi['slang'] == kata, 'formal'].values[0]
                    ulasan_normalisasi.append(formal)
                else:
                    ulasan_normalisasi.append(kata)
            return ulasan_normalisasi

        Fact = StemmerFactory()
        Stemmer = Fact.create_stemmer()
        
        def stemming(ulasan):
            result = []
            for word in ulasan:
                result.append(Stemmer.stem(word))
            return result

        def remove_stopword(ulasan):
            result = []
            for word in ulasan:
                if word not in stopwords.words('indonesian'):
                    result.append(word)
            return result

        # Preprocessing data training
        data['case_folding'] = data['ulasan'].apply(case_fold)
        data['clean'] = data['case_folding'].apply(remove_punctuation)
        data['tokenisasi'] = data['clean'].apply(tokenize)
        data['normal'] = data['tokenisasi'].apply(normalization)
        data['stemming'] = data['normal'].apply(stemming)
        data['stopword'] = data['stemming'].apply(remove_stopword)
        data['final'] = data['stopword'].apply(lambda tokens: ' '.join(tokens))

        # Siapkan data untuk training (menggunakan pendekatan XGBoost)
        X = data['final'].values.tolist()
        
        # Asumsikan label ada di kolom 1-10 seperti code asli
        # PERBAIKAN: Konversi tipe data dan pastikan format yang benar
        y_raw = data[data.columns[1:11]].values
        
        # Konversi ke integer dan pastikan dalam format yang benar untuk scikit-multilearn
        y = y_raw.astype(np.int32)
        
        # Pastikan tidak ada nilai yang hilang
        if np.any(pd.isna(y)):
            st.error("Terdapat nilai yang hilang dalam data label. Silakan periksa data Anda.")
            st.stop()

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=2500, max_df=0.9)
        X_tfidf = vectorizer.fit_transform(X)
        
        # Split data untuk training
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.20, random_state=42)
        
        # PERBAIKAN: Pastikan y_train dalam format yang benar
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)
        
        # Model XGBoost dengan skenario terbaik (skenario 1)
        model = LabelPowerset(
            XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                n_estimators=100,
                learning_rate=0.1,
                reg_lambda=1,
                gamma=0,
                random_state=42
            )
        )
        
        # Training model dengan error handling
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            st.error(f"Error dalam training model: {str(e)}")
            st.stop()

        # Preprocessing input user
        df_mentah['case_folding'] = df_mentah['Ulasan'].apply(case_fold)
        df_mentah['clean'] = df_mentah['case_folding'].apply(remove_punctuation)
        df_mentah['tokenisasi'] = df_mentah['clean'].apply(tokenize)
        df_mentah['normal'] = df_mentah['tokenisasi'].apply(normalization)
        df_mentah['stemming'] = df_mentah['normal'].apply(stemming)
        df_mentah['stopword'] = df_mentah['stemming'].apply(remove_stopword)
        df_mentah['final'] = df_mentah['stopword'].apply(lambda tokens: ' '.join(tokens))

        # Transform input user dengan vectorizer yang sama
        transform = vectorizer.transform(df_mentah['final'])

        # Prediksi menggunakan XGBoost model
        try:
            pred = model.predict(transform)
        except Exception as e:
            st.error(f"Error dalam prediksi: {str(e)}")
            st.stop()

        st.write("**Aspek dan sentimen yang terkandung dalam ulasan**")
        
        nama_fitur = [['Makanan', 'Positif'], ['Makanan', 'Negatif'], ['Layanan', 'Positif'], ['Layanan', 'Negatif'], ['Tempat', 'Positif'], ['Tempat', 'Negatif'], ['Harga', 'Positif'], ['Harga', 'Negatif'], ['Lainnya', 'Positif'], ['Lainnya', 'Negatif']]

        # Dictionary untuk warna background berdasarkan aspek
        aspek_warna = {
            'Makanan': '#fecdd3',  # pink
            'Layanan': '#fde68a',  # kuning
            'Tempat': '#a7f3d0',   # hijau
            'Harga': '#e9d5ff',    # ungu
            'Lainnya': '#e5e7eb'   # abu-abu
        }

        sentimen_warna = {
            'Positif': '#93c5fd',  # biru
            'Negatif': '#fca5a5'   # merah
        }

        for i in range(len(pred)):
            # PERBAIKAN: Pastikan pred dalam format yang benar
            if hasattr(pred, 'toarray'):
                pred_array = pred.toarray()
            else:
                pred_array = pred
                
            indeks_positif = np.where(pred_array[i] == 1)[0]
            fitur_positif = [nama_fitur[idx] for idx in indeks_positif]
            
            if len(fitur_positif) > 0:
                fitur_positif_str = [
                    f"Aspek: <span style='background-color: {aspek_warna[fitur[0]]}; border-radius:5px; padding: 2px 4px; font-weight: 600;'>{fitur[0]}</span> "
                    f"Sentimen: <span style='background-color: {sentimen_warna[fitur[1]]}; border-radius:5px; padding: 2px 4px; font-weight: 600;'>{fitur[1]}</span>"
                    for fitur in fitur_positif
                ]
                st.markdown("<br>".join(fitur_positif_str), unsafe_allow_html=True)
            else:
                st.write("Tidak ada aspek atau sentimen yang terdeteksi dalam ulasan ini.")
