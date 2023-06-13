import streamlit as st
import pandas as pd
import numpy as np
import string
from itertools import chain
from tqdm.auto import tqdm
import re, string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

st.title('Prediksi tweet covid 19')
text = st.text_input("Masukkan teks")
button=st.button('Hasil Prediksi')

if button:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    # Download resources
    nltk.download('punkt')
    nltk.download('stopwords')

    # Load dataset
    df = pd.read_csv('https://github.com/davata1/pba/blob/main/covid.csv')
    df.drop_duplicates(inplace=True)
    df = df.drop(['Unnamed:','Datetime', 'Tweet Id'], axis=1)  
    # Text Cleaning
    def cleaning(text):
        text = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});').sub('', str(text))
        text = text.lower()
        text = text.strip()
        text = re.compile('<.*?>').sub('', text)
        text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
        text = re.sub('\s+', ' ', text)
        text = re.sub(r'\[[0-9]*\]', ' ', text)
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
        text = re.sub(r'\d', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub('nan', '', text)
        return text

    def preprocess_data(df):
        #Removing stopwords
        stop_words = set(chain(stopwords.words('indonesian'), stopwords.words('english')))
        df['Text_token'] = df['Text_token'].apply(lambda x: [w for w in x if not w in stop_words])

    # Tokenizing text
        df['Text_token'] = df['Text'].apply(lambda x: word_tokenize(x))

    # Stemming text
        tqdm.pandas()
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        df['Text_token'] = df['Text_token'].progress_apply(lambda x: stemmer.stem(' '.join(x)).split(' '))           

        # Mengambil input teks dari pengguna
        #st.write("Hasil Preprocessing:")
        analisis=cleaning(text)
        #st.write(analisis)
                
        import pickle
        with open ('modelKNNrill.pkl', 'rb') as r:
            knn=pickle.load(r)
            import pickle
            with open('tfidf.pkl', 'rb') as f:
                tfidf= pickle.load(f)
                
                hasil=tfidf.transform([analisis])
                predictions = knn.predict(hasil)
                for a in predictions:
                    st.write('Text : ',analisis)
                    st.write('Sentimen :', a)

            def process(df):
                return preprocess_data(df)
                