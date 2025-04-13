import pickle
import pandas as pd
import numpy as np
import re
import string
import nltk
nltk.download('punkt')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import streamlit as st

# Define powerset label mapping
powerset_labels = {
    1: {"Makanan": "Positif"},
    2: {"Makanan": "Negatif"},
    3: {"Pelayanan": "Positif"},
    4: {"Pelayanan": "Negatif"},
    5: {"Tempat": "Positif"},
    6: {"Tempat": "Negatif"},
    7: {"Harga": "Positif"},
    8: {"Harga": "Negatif"},
    9: {"Makanan": "Positif", "Pelayanan": "Positif"},
    10: {"Makanan": "Positif", "Pelayanan": "Negatif"},
    11: {"Makanan": "Positif", "Tempat": "Positif"},
    12: {"Makanan": "Positif", "Tempat": "Negatif"},
    13: {"Makanan": "Positif", "Harga": "Positif"},
    14: {"Makanan": "Positif", "Harga": "Negatif"},
    15: {"Makanan": "Negatif", "Pelayanan": "Negatif"},
    16: {"Makanan": "Negatif", "Tempat": "Positif"},
    17: {"Makanan": "Negatif", "Tempat": "Negatif"},
    18: {"Makanan": "Negatif", "Harga": "Positif"},
    19: {"Makanan": "Negatif", "Harga": "Negatif"},
    20: {"Pelayanan": "Positif", "Tempat": "Positif"},
    21: {"Pelayanan": "Negatif", "Tempat": "Positif"},
    22: {"Pelayanan": "Negatif", "Tempat": "Negatif"},
    23: {"Pelayanan": "Negatif", "Harga": "Positif"},
    24: {"Pelayanan": "Negatif", "Harga": "Negatif"},
    25: {"Makanan": "Positif", "Pelayanan": "Positif", "Tempat": "Positif"},
    26: {"Makanan": "Positif", "Pelayanan": "Positif", "Tempat": "Negatif"},
    27: {"Makanan": "Positif", "Pelayanan": "Negatif", "Tempat": "Positif"},
    28: {"Makanan": "Positif", "Pelayanan": "Negatif", "Tempat": "Negatif"},
    29: {"Makanan": "Positif", "Pelayanan": "Negatif", "Harga": "Positif"},
    30: {"Makanan": "Positif", "Tempat": "Positif", "Harga": "Positif"},
    31: {"Makanan": "Negatif", "Pelayanan": "Negatif", "Tempat": "Positif"},
    32: {"Makanan": "Negatif", "Pelayanan": "Negatif", "Tempat": "Negatif"},
    33: {"Pelayanan": "Positif", "Tempat": "Positif", "Harga": "Negatif"},
    34: {"Pelayanan": "Negatif", "Tempat": "Positif", "Harga": "Negatif"},
    35: {"Pelayanan": "Negatif", "Tempat": "Negatif", "Harga": "Negatif"},
    36: {"Makanan": "Positif", "Pelayanan": "Positif", "Tempat": "Positif", "Harga": "Positif"},
    37: {"Makanan": "Positif", "Tempat": "Positif", "Tempat": "Negatif"},
    38: {"Makanan": "Negatif", "Tempat": "Positif", "Tempat": "Negatif"},
    39: {"Makanan": "Negatif", "Tempat": "Positif", "Harga": "Positif"},
    40: {"Makanan": "Positif", "Makanan": "Negatif"},
    41: {"Pelayanan": "Positif", "Pelayanan": "Negatif"},
    42: {"Tempat": "Positif", "Tempat": "Negatif"},
    43: {"Makanan": "Positif", "Makanan": "Negatif", "Pelayanan": "Negatif"},
    44: {"Makanan": "Positif", "Makanan": "Negatif", "Tempat": "Negatif"},
    45: {"Pelayanan": "Positif", "Tempat": "Positif", "Tempat": "Negatif"},
    46: {"Makanan": "Positif", "Makanan": "Negatif", "Pelayanan": "Positif", "Pelayanan": "Negatif", "Tempat": "Positif"}
}

with st.container():
    st.title('Aplikasi Analisis Sentimen Berbasis Aspek :red[Ulasan Rumah Makan Bebek Sinjay]')
    input = st.text_area("**Masukkan Ulasan**")
    submit = st.button("Proses", type="primary")

    if submit:
        df_mentah = pd.DataFrame({'Ulasan': [input]})

        data = pd.read_excel('data fix.xlsx')

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

        kamus_normalisasi = pd.read_csv('colloquial-indonesian-lexicon.csv')
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

        # Preprocessing pada data training
        data['case_folding'] = data['ulasan'].apply(case_fold)
        data['clean'] = data['case_folding'].apply(remove_punctuation)
        data['tokenisasi'] = data['clean'].apply(tokenize)
        data['normal'] = data['tokenisasi'].apply(normalization)
        data['stemming'] = data['normal'].apply(stemming)
        data['stopword'] = data['stemming'].apply(remove_stopword)
        data['final'] = data['stopword'].apply(lambda tokens: ' '.join(tokens))

        # Asumsikan label powerset ada di kolom 'label'
        X = data['final'].values.tolist()
        
        # Vectorize menggunakan TF-IDF
        vectorizer = TfidfVectorizer(max_features=2500, max_df=0.9)
        tfidf_vectors = vectorizer.fit_transform(X)
        
        # Load model yang sudah dilatih untuk powerset classification
        with open('model_clf_powerset.pkl', 'rb') as r:
            load_model = pickle.load(r)

        # Preprocessing pada data input
        df_mentah['case_folding'] = df_mentah['Ulasan'].apply(case_fold)
        df_mentah['clean'] = df_mentah['case_folding'].apply(remove_punctuation)
        df_mentah['tokenisasi'] = df_mentah['clean'].apply(tokenize)
        df_mentah['normal'] = df_mentah['tokenisasi'].apply(normalization)
        df_mentah['stemming'] = df_mentah['normal'].apply(stemming)
        df_mentah['stopword'] = df_mentah['stemming'].apply(remove_stopword)
        df_mentah['final'] = df_mentah['stopword'].apply(lambda tokens: ' '.join(tokens))

        # Transform input text
        transform = vectorizer.transform(df_mentah['final'])
        transform_dense = transform.toarray()

        # Predict menggunakan model
        pred_label = load_model.predict(transform_dense)[0]
        
        # Dictionary untuk warna background berdasarkan aspek
        aspek_warna = {
            'Makanan': '#fecdd3',  # pink
            'Pelayanan': '#fde68a',  # kuning
            'Tempat': '#a7f3d0',   # hijau
            'Harga': '#e9d5ff',    # ungu
        }

        sentimen_warna = {
            'Positif': '#93c5fd',  # biru
            'Negatif': '#fca5a5'   # merah
        }

        # Menampilkan hasil prediksi
        st.write("**Aspek dan sentimen yang terkandung dalam ulasan**")
        
        if pred_label in powerset_labels:
            aspects_sentiments = powerset_labels[pred_label]
            
            fitur_positif_str = []
            for aspek, sentimen in aspects_sentiments.items():
                fitur_positif_str.append(
                    f"Aspek: <span style='background-color: {aspek_warna.get(aspek, '#e5e7eb')}; border-radius:5px; padding: 2px 4px; font-weight: 600;'>{aspek}</span> "
                    f"Sentimen: <span style='background-color: {sentimen_warna.get(sentimen, '#e5e7eb')}; border-radius:5px; padding: 2px 4px; font-weight: 600;'>{sentimen}</span> "
                )
            
            st.markdown("<br>".join(fitur_positif_str), unsafe_allow_html=True)
        else:
            st.warning(f"Tidak dapat mengidentifikasi aspek dan sentimen dari ulasan (Label {pred_label} tidak terdefinisi)")

# Kode untuk melatih model powerset (tidak ditampilkan di aplikasi Streamlit)
def train_powerset_model():
    # Load data
    data = pd.read_excel('data fix.xlsx')
    
    # Preprocessing
    data['case_folding'] = data['ulasan'].apply(case_fold)
    data['clean'] = data['case_folding'].apply(remove_punctuation)
    data['tokenisasi'] = data['clean'].apply(tokenize)
    data['normal'] = data['tokenisasi'].apply(normalization)
    data['stemming'] = data['normal'].apply(stemming)
    data['stopword'] = data['stemming'].apply(remove_stopword)
    data['final'] = data['stopword'].apply(lambda tokens: ' '.join(tokens))
    
    # Assuming 'label' is the column with powerset labels
    X = data['final'].values.tolist()
    y = data['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=2500, max_df=0.9)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train_vec, y_train)
    
    # Save model and vectorizer
    with open('model_clf_powerset.pkl', 'wb') as f:
        pickle.dump(clf, f)
    
    with open('vectorizer_powerset.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Evaluate
    accuracy = clf.score(X_test_vec, y_test)
    print(f"Accuracy: {accuracy}")
    
    return clf, vectorizer