import streamlit as st
import pandas as pd
import numpy as np
import string
from sklearn.pipeline import Pipeline
import re
import scipy.sparse as sp

st.title('Prediksi tweet covid 19')
text = st.text_input("Masukkan teks")
button=st.button('Hasil Prediiksi')

if button :
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import PorterStemmer
        import re

        # Menginisialisasi Streamlit
        #st.title("Preprocessing pada Teks"

        # Mengaktifkan resource NLTK yang diperlukan
        nltk.download('punkt')
        nltk.download('stopwords')
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        
        # Membaca kamus dari file Excel
        df = pd.read_csv('https://github.com/davata1/pba/blob/main/covid.csv')
        # Mengubah kamus menjadi dictionary
        kamus_dict = dict(zip(kamus_df['before'], kamus_df['after']))
        def normalize_typo(text):
            words = text.split()
            normalized_words = []
            for word in words:
                if word in kamus_dict:
                    corrected_word = kamus_dict[word]
                    normalized_words.append(corrected_word)
                else:
                    normalized_words.append(word)
            normalized_text = ' '.join(normalized_words)
            return normalized_text
        
        # Mendefinisikan fungsi pra-pemrosesan
        def preprocess_text(text):
            # Menghilangkan karakter yang tidak diinginkan
            text = text.strip(" ")
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
            text = re.sub(r'[?|$|.|!_:")(-+,]', ' ', text)
            text = re.sub(r'\d+', ' ', text)
            text = re.sub(r"\b[a-zA-Z]\b", " ",text)
            text = re.sub('\s+',' ', text)
            text = normalize_typo(text)
            # Tokenisasi teks menjadi kata-kata
            tokens = word_tokenize(text)
            
            # Menghapus kata-kata yang tidak bermakna (stopwords)
            stop_words = set(stopwords.words('Indonesian'))
            tokens = [token for token in tokens if token not in stop_words]
            
            # Menggabungkan kata-kata kembali menjadi teks yang telah dipreprocessed
            processed_text = ' '.join(tokens)
            
            # Melakukan stemming pada teks menggunakan PySastrawi
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            stemmed_text = stemmer.stem(processed_text)
            return stemmed_text

        # Mengambil input teks dari pengguna
        #st.write("Hasil Preprocessing:")
        analisis=preprocess_text(text)
        #st.write(analisis)
        
        import pickle
        with open ('modelKNNrill.pkl', 'rb') as r:
            asknn=pickle.load(r)
        import pickle
        with open('tfidf.pkl', 'rb') as f:
            vectoriz= pickle.load(f)    
        
        
        hastfidf=vectoriz.transform([analisis])
        predictions = asknn.predict(hastfidf)
        for i in predictions:
            st.write('Text : ',analisis)
            st.write('Sentimenm :', i)
        #Menampilkan hasil prediksi
        #sentiment = asknn.predict(cosim)
        #st.write("Sentimen:", sentiment)


    
