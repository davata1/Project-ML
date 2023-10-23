import streamlit as st
import pandas as pd
import numpy as np

##Abstrak tokens
df = pd.read_csv('https://raw.githubusercontent.com/davata1/ppw/main/DataPTAInformatikaLabel.csv',delimiter=';')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.dropna(inplace=True)
import re, string

# Text Cleaning
def cleaning(text):
    # Menghapus tag HTML
    text = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});').sub('', str(text))

    # Mengubah seluruh teks menjadi huruf kecil
    text = text.lower()

    # Menghapus spasi pada teks
    text = text.strip()

    # Menghapus Tanda Baca, karakter spesial, and spasi ganda
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub("â", "", text)

    # Menghapus Nomor
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Mengubah text yang berisi 'nan' dengan whitespace agar nantinya dapat dihapus
    text = re.sub('nan', '', text)

    return text

#Membuat abstrak
df['Abstrak'] = df['Abstrak'].apply(lambda x: cleaning(x))

import nltk
from nltk.tokenize import word_tokenize
nltk.download('popular')

df['abstrak_tokens'] = df['Abstrak'].apply(lambda x: word_tokenize(x))


st.header("UTS PPW")
st.subheader("Mengambil Data CSV pada Github")
st.text("load data(DataStemming.csv) csv yang sudah berhasil di stemming")

##Load data
df = pd.read_csv('https://raw.githubusercontent.com/davata1/Dataset/main/DataSteamingg.csv')
df.head()

st.text("Ektraksi Fitur")
df['Abstrak']

st.text("Term Frekuensi")
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()

# Gantilah nilai NaN dalam kolom 'Abstrak' dengan string kosong
df['Abstrak'].fillna('', inplace=True)

X_count = count_vectorizer.fit_transform(np.array(df['Abstrak']))

terms_count = count_vectorizer.get_feature_names_out()
df_countvect = pd.DataFrame(data=X_count.toarray(), columns=terms_count)
df_countvect

token_counts = df_countvect.sum(axis=0)

non_zero_token_counts = token_counts[token_counts != 0]

st.text("One Hot Encoding")
df_binary = df_countvect.applymap(lambda x: 1 if x > 0 else 0)
df_binary

st.text("TF IDF")
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['Abstrak'].tolist())

terms = vectorizer.get_feature_names_out()
df_tfidfvect = pd.DataFrame(data=X_tfidf.toarray(), columns=terms)
df_tfidfvect

st.text("Log Frekuensi")
df_log = df_countvect.applymap(lambda x: np.log1p(x) if x > 0 else 0)
df_log

st.text("LDA Model")
import gensim
from gensim import corpora
from gensim.models import LdaModel
import pandas as pd
import matplotlib.pyplot as plt

# Ubah teks ke dalam format yang cocok untuk Gensim
documents =df['abstrak_tokens']

# Membuat kamus (dictionary) dari kata-kata unik dalam dokumen
dictionary = corpora.Dictionary(documents)

# Membuat korpus (bag-of-words) dari dokumen
corpus = [dictionary.doc2bow(doc) for doc in documents]

# Melatih model LDA
lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=30)

# Membuat DataFrame untuk menampilkan proporsi topik dalam dokumen
document_topic_df = pd.DataFrame()

for doc in corpus:
    topic_distribution = lda_model.get_document_topics(doc, minimum_probability=0)
    doc_topic_props = {} #mengubah tampilan agar topik di probalility hilang dan ada pada tabel diatasnya
    for topic_id, prob in topic_distribution:
        key = f"Topik {topic_id + 1}"
        doc_topic_props[key] = prob
    # doc_topic_props["Judul"] = datajudul
    document_topic_df = pd.concat([document_topic_df, pd.Series(doc_topic_props)], ignore_index=True, axis=1)

document_topic_df = document_topic_df.transpose()  # Transpose agar topik menjadi kolom

column_names = [f"Topik {i + 1}" for i in range(lda_model.num_topics)]
document_topic_df.columns = column_names

# Menampilkan tabel proporsi topik dalam dokumen
st.text("Tabel Proporsi Topik dalam Dokumen:")
document_topic_df

# Membuat DataFrame untuk menampilkan proporsi kata dalam topik
topic_word_df = pd.DataFrame()

for topic_id in range(lda_model.num_topics):
    topic_words = lda_model.show_topic(topic_id, topn=10)  # Ambil 10 kata kunci teratas
    words_list = [word for word, _ in topic_words]
    topic_word_df[f"Topik {topic_id + 1}"] = words_list

# Menampilkan tabel proporsi kata dalam topik
st.text("\nTabel Proporsi Kata dalam Topik:")
st.write(topic_word_df)
