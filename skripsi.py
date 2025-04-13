import pickle
import pandas as pd
import numpy as np
import re
import string
import nltk
import streamlit as st
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import requests
from io import BytesIO

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords

# Load model function with error handling
@st.cache_resource
def load_pickle_model():
    try:
        # First try to load local model
        if os.path.exists('model_clf.pkl'):
            with open('model_clf.pkl', 'rb') as r:
                return pickle.load(r)
        else:
            # If not available locally, try to download
            model_url = "https://github.com/davata1/Project-ML/raw/main/model_clf.pkl"
            response = requests.get(model_url)
            if response.status_code == 200:
                return pickle.loads(response.content)
            else:
                st.error("Failed to load model from GitHub. Please check your internet connection.")
                return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load data function with error handling
@st.cache_data
def load_data():
    try:
        return pd.read_excel('https://github.com/davata1/Project-ML/raw/refs/heads/main/data%20fix.xlsx')
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load normalization dictionary
@st.cache_data
def load_normalization_dict():
    try:
        kamus = pd.read_csv('https://github.com/davata1/Project-ML/raw/refs/heads/main/colloquial-indonesian-lexicon.csv')
        return kamus.drop(columns=['In-dictionary','context','category1','category2', 'category3'])
    except Exception as e:
        st.error(f"Error loading normalization dictionary: {str(e)}")
        return None

# Text preprocessing functions
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
    if isinstance(text, str):
        return nltk.word_tokenize(text)
    return []

def normalization(token, kamus_normalisasi):
    ulasan_normalisasi = []
    for kata in token:
        if kata in kamus_normalisasi['slang'].values:
            formal = kamus_normalisasi.loc[kamus_normalisasi['slang'] == kata, 'formal'].values[0]
            ulasan_normalisasi.append(formal)
        else:
            ulasan_normalisasi.append(kata)
    return ulasan_normalisasi

# Create stemmer
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

# Main application
def main():
    with st.container():
        st.title('Aplikasi Analisis Sentimen Berbasis Aspek :red[Ulasan Rumah Makan Bebek Sinjay]')
        input_text = st.text_area("**Masukkan Ulasan**")
        submit = st.button("Proses", type="primary")

        if submit:
            if not input_text.strip():
                st.warning("Mohon masukkan ulasan terlebih dahulu.")
                return
                
            try:
                # Create dataframe from input
                df_mentah = pd.DataFrame({'Ulasan': [input_text]})
                
                # Load data and resources
                data = load_data()
                kamus_normalisasi = load_normalization_dict()
                model = load_pickle_model()
                
                if data is None or kamus_normalisasi is None or model is None:
                    return
                    
                # Process training data
                data['case_folding'] = data['ulasan'].apply(case_fold)
                data['clean'] = data['case_folding'].apply(remove_punctuation)
                data['tokenisasi'] = data['clean'].apply(tokenize)
                data['normal'] = data['tokenisasi'].apply(lambda x: normalization(x, kamus_normalisasi))
                data['stemming'] = data['normal'].apply(stemming)
                data['stopword'] = data['stemming'].apply(remove_stopword)
                data['final'] = data['stopword'].apply(lambda tokens: ' '.join(tokens))

                X = data['final'].values.tolist()
                
                # Vectorize training data
                vectorizer = TfidfVectorizer(max_features=2500, max_df=0.9)
                tfidf_vectors = vectorizer.fit_transform(X)
                tf_idf_array = tfidf_vectors.toarray()
                corpus = vectorizer.get_feature_names_out()
                hasil_tfidf = pd.DataFrame(tf_idf_array, columns=corpus)
                
                # Process input data
                df_mentah['case_folding'] = df_mentah['Ulasan'].apply(case_fold)
                df_mentah['clean'] = df_mentah['case_folding'].apply(remove_punctuation)
                df_mentah['tokenisasi'] = df_mentah['clean'].apply(tokenize)
                df_mentah['normal'] = df_mentah['tokenisasi'].apply(lambda x: normalization(x, kamus_normalisasi))
                df_mentah['stemming'] = df_mentah['normal'].apply(stemming)
                df_mentah['stopword'] = df_mentah['stemming'].apply(remove_stopword)
                df_mentah['final'] = df_mentah['stopword'].apply(lambda tokens: ' '.join(tokens))
                
                # Transform input
                transform = vectorizer.transform(df_mentah['final'])
                transform_dense = np.asarray(transform.todense())
                
                # Make prediction
                pred = model.predict(transform_dense)
                
                # Display results
                st.write("**Aspek dan sentimen yang terkandung dalam ulasan**")
                
                nama_fitur = [
                    ['Makanan','Positif'],['Makanan','Negatif'],
                    ['Layanan','Positif'],['Layanan','Negatif'],
                    ['Tempat','Positif'],['Tempat','Negatif'],
                    ['Harga','Positif'],['Harga','Negatif'],
                    ['Lainnya','Positif'],['Lainnya','Negatif']
                ]
                
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
                    indeks_positif = np.where(pred[i] == 1)[0]
                    if len(indeks_positif) == 0:
                        st.write("Tidak ada aspek dan sentimen yang terdeteksi dalam ulasan ini.")
                    else:
                        fitur_positif = [nama_fitur[idx] for idx in indeks_positif]
                        fitur_positif_str = [
                            f"Aspek: <span style='background-color: {aspek_warna[fitur[0]]}; border-radius:5px; padding: 2px 4px; font-weight: 600;'>{fitur[0]}</span> "
                            f"Sentimen: <span style='background-color: {sentimen_warna[fitur[1]]}; border-radius:5px; padding: 2px 4px; font-weight: 600;'>{fitur[1]}</span> "
                            for fitur in fitur_positif
                        ]
                        st.markdown("<br>".join(fitur_positif_str), unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Terjadi kesalahan dalam pemrosesan: {str(e)}")

if __name__ == "__main__":
    main()
