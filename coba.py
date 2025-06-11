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

        data = pd.read_excel('https://github.com/davata1/Project-ML/raw/refs/heads/main/sinjaymadura.xlsx')

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

        kamus_normalisasi = pd.read_csv('https://github.com/davata1/Project-ML/raw/refs/heads/main/colloquial-indonesian-lexicon.csv')
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

        # Siapkan data untuk training
        X = data['final'].values.tolist()
        
        # Konversi label powerset menggunakan MultiLabelBinarizer
        st.write("**Mengkonversi Label Powerset dengan MultiLabelBinarizer...**")
        
        # Cek struktur data
        st.write(f"Kolom data: {data.columns.tolist()}")
        st.write(f"Sample label powerset: {data.iloc[:5, 1].tolist()}")
        
        # Split label jadi list (sama seperti di Google Colab)
        data['label_split'] = data.iloc[:, 1].apply(lambda x: str(x).split(',') if pd.notna(x) and x != '' else [])
        
        # Bersihkan whitespace
        data['label_split'] = data['label_split'].apply(lambda x: [label.strip() for label in x if label.strip()])
        
        # MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(data['label_split'])
        
        st.success(f"Konversi berhasil! Shape label: {y.shape}")
        st.write(f"Classes yang ditemukan: {mlb.classes_}")
        st.write(f"Jumlah classes: {len(mlb.classes_)}")
        st.write(f"Distribusi label (jumlah 1 per kolom): {np.sum(y, axis=0)}")
        
        # Tampilkan contoh hasil konversi
        st.write("**Contoh Hasil Konversi:**")
        for i in range(min(3, len(data))):
            original_label = data.iloc[i, 1]
            label_list = data['label_split'].iloc[i]
            binary_result = y[i]
            st.write(f"- '{original_label}' → {label_list} → {binary_result}")
        
        # Simpan mlb untuk digunakan nanti dalam prediksi (opsional)
        # Ini berguna jika ingin menampilkan nama label hasil prediksi

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
        
        # Menggunakan classes dari MultiLabelBinarizer
        classes = mlb.classes_
        
        # Dictionary untuk warna background berdasarkan aspek
        aspek_warna = {
            'makanan positif': '#fecdd3',    # pink
            'makanan negatif': '#fca5a5',    # merah muda
            'layanan positif': '#fde68a',    # kuning
            'layanan negatif': '#f59e0b',    # orange
            'pelayanan positif': '#fde68a',  # kuning (alias layanan)
            'pelayanan negatif': '#f59e0b',  # orange (alias layanan)
            'tempat positif': '#a7f3d0',     # hijau
            'tempat negatif': '#10b981',     # hijau tua
            'harga positif': '#e9d5ff',      # ungu
            'harga negatif': '#8b5cf6',      # ungu tua
            'lainnya positif': '#e5e7eb',    # abu-abu
            'lainnya negatif': '#6b7280'     # abu-abu tua
        }

        for i in range(len(pred)):
            # Pastikan pred dalam format yang benar
            if hasattr(pred, 'toarray'):
                pred_array = pred.toarray()
            else:
                pred_array = pred
                
            indeks_positif = np.where(pred_array[i] == 1)[0]
            
            if len(indeks_positif) > 0:
                fitur_positif_str = []
                for idx in indeks_positif:
                    if idx < len(classes):
                        label_name = classes[idx]
                        # Tentukan warna berdasarkan label
                        warna = aspek_warna.get(label_name, '#e5e7eb')  # default abu-abu
                        
                        fitur_positif_str.append(
                            f"<span style='background-color: {warna}; border-radius:5px; padding: 4px 8px; margin: 2px; font-weight: 600; display: inline-block;'>{label_name}</span>"
                        )
                
                if fitur_positif_str:
                    st.markdown(" ".join(fitur_positif_str), unsafe_allow_html=True)
                else:
                    st.write("Tidak ada aspek atau sentimen yang terdeteksi dalam ulasan ini.")
            else:
                st.write("Tidak ada aspek atau sentimen yang terdeteksi dalam ulasan ini.")
