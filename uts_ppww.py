import streamlit as st
import pandas as pd
import numpy as np

st.header("UTS PPW")
st.subheader("Mengambil Data CSV pada Github")
st.text("load data(DataStemming.csv) csv yang sudah berhasil di stemming")

##Load data
df = pd.read_csv('https://raw.githubusercontent.com/davata1/ppw/main/DataSteaming.csv')
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
df_countvect = pd.DataFrame(data = X_count.toarray(),columns = terms_count)
df_countvect

token_counts = df_countvect.sum(axis=0)

non_zero_token_counts = token_counts[token_counts != 0]

# print("Token Counts yang Tidak Sama dengan 0:")
# print(non_zero_token_counts)

st.text("One Hot Encoding")
df_binary = df_countvect.applymap(lambda x: 1 if x > 0 else 0)
df_binary

st.text("TF IDF")
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['Abstrak'].tolist())

terms = vectorizer.get_feature_names_out()
df_tfidfvect = pd.DataFrame(data = X_tfidf.toarray(),columns = terms)
df_tfidfvect

st.text("Log Frekuensi")
df_log = df_countvect.applymap(lambda x: np.log1p(x) if x > 0 else 0)
df_log

st.text("LDA Model")
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

lda_model = LatentDirichletAllocation(n_components=3, doc_topic_prior=0.1, topic_word_prior=0.2, random_state=42)
lda_model.fit(df_countvect)

lda_model

doc_topic_proportions = lda_model.transform(df_countvect)

for i, doc in enumerate(df['Abstrak']):
    # print(f"Dokumen {i+1}:")
    for j, topic_prob in enumerate(doc_topic_proportions[i]):
        # print(f"Topik {j+1}: {topic_prob:.4f}")
    # print()

        topic_word_distributions = lda_model.components_

        feature_names = vectorizer.get_feature_names_out()

for topic_idx, topic in enumerate(topic_word_distributions):
    top_words_idx = topic.argsort()[::-1][:10]  # Ambil 10 kata teratas
    top_words = [feature_names[i] for i in top_words_idx]
    st.text(f"Topik {topic_idx+1}:")
    st.text(", ".join(top_words))
    st.text("")