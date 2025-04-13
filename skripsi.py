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
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
import streamlit as st

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
      data=re.sub('@[^\s]+', ' ', text)
      data = re.sub(r'http\S*', ' ', data)
      data=data.translate(str.maketrans(' ',' ',string.punctuation))
      data=re.sub('[^a-zA-Z]',' ',data)
      data=re.sub("\n"," ",data)
      data=re.sub(r"\b[a-zA-z]\b"," ",data)
      return data

    def tokenize (text):
      return nltk.word_tokenize(text)

    kamus_normalisasi = pd.read_csv('colloquial-indonesian-lexicon.csv')
    kamus_normalisasi = kamus_normalisasi.drop(columns=['In-dictionary','context','category1','category2', 'category3'])

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

    data['case_folding']=data['ulasan'].apply(case_fold)
    data['clean']= data['case_folding'].apply(remove_punctuation)
    data['tokenisasi'] = data['clean'].apply(tokenize)
    data['normal'] = data['tokenisasi'].apply(normalization)
    data['stemming'] = data['normal'].apply(stemming)
    data['stopword'] = data['stemming'].apply(remove_stopword)
    data['final'] = data['stopword'].apply(lambda tokens: ' '.join(tokens))

    X = data['final'].values.tolist()
    y = np.asarray(data[data.columns[1:11]])

    X = data['final'].values.tolist()

    vectorizer = TfidfVectorizer(max_features=2500, max_df=0.9)
    tfidf_vectors = vectorizer.fit_transform(X)
    tf_idf_array = tfidf_vectors.toarray()

    corpus = vectorizer.get_feature_names_out()

    hasil_tfidf = pd.DataFrame(tf_idf_array, columns=corpus)

    with open ('model_clf.pkl', 'rb') as r:
      load_model = pickle.load(r)

    df_mentah['case_folding']=df_mentah['Ulasan'].apply(case_fold)
    df_mentah['clean']= df_mentah['case_folding'].apply(remove_punctuation)
    df_mentah['tokenisasi'] = df_mentah['clean'].apply(tokenize)
    df_mentah['normal'] = df_mentah['tokenisasi'].apply(normalization)
    df_mentah['stemming'] = df_mentah['normal'].apply(stemming)
    df_mentah['stopword'] = df_mentah['stemming'].apply(remove_stopword)
    df_mentah['final'] = df_mentah['stopword'].apply(lambda tokens: ' '.join(tokens))
    # st.dataframe(df_mentah['final'])

    transform = vectorizer.transform(df_mentah['final'])

    transform_dense = np.asarray(transform.todense())

    pred = load_model.predict(transform_dense)
    y_pred = load_model.predict(hasil_tfidf)

    # st.text(pred)
    st.write("**Aspek dan sentimen yang terkandung dalam ulasan**")
    
    nama_fitur = [['Makanan','Positif'],['Makanan','Negatif'],['Layanan','Positif'],['Layanan','Negatif'],['Tempat','Positif'],['Tempat','Negatif'],['Harga','Positif'],['Harga','Negatif'],['Lainnya','Positif'],['Lainnya','Negatif']]

    # Dictionary untuk warna background berdasarkan aspek
    aspek_warna = {
        'Makanan': '#fecdd3',  # pink
        'Layanan': '#fde68a',  # biru
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
        fitur_positif = [nama_fitur[idx] for idx in indeks_positif]
        fitur_positif_str = [
            f"Aspek: <span style='background-color: {aspek_warna[fitur[0]]}; border-radius:5px; padding: 2px 4px; font-weight: 600;'>{fitur[0]}</span> "
            f"Sentimen: <span style='background-color: {sentimen_warna[fitur[1]]}; border-radius:5px; padding: 2px 4px; font-weight: 600;'>{fitur[1]}</span> "
            for fitur in fitur_positif
        ]
        st.markdown("<br>".join(fitur_positif_str), unsafe_allow_html=True)