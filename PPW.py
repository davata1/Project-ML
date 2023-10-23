import streamlit as st
import pandas as pd
import numpy as np
from sequence_split import split_sequence
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.model_selection import train_test_split
import pickle
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import os
# path_to_fungsi = r'A:\Matkul\Semester 7\PPW\UTS\Fungsi'
# sys.path.append("A:/Matkul/Semester 7/PPW/UTS/Fungsi")
from Cleaning import cleaning
from tokenisasi import tokenize_text
from stopword import remove_stopwords
from steaming import stem_text

#import data
data = pd.read_csv('https://raw.githubusercontent.com/davata1/ppw/main/DataPTAInformatikaLabel.csv',delimiter=';')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]


def main():
    st.sidebar.title("Pengolahan Data PTA Trunojoyo")
    menu = ["Data", "Pre processing data","Ekstraksi Fitur","Skip Gram","Topic Modeling", "Klasifikasi Data"]
    choice = st.sidebar.selectbox("Menu", menu)
    global data

    #menampilkan Data
    if choice == "Data":
        st.title("Pengolahan Data PTA Trunojoyo")
        st.markdown("<h4>Menampilkan Data</h4>", unsafe_allow_html=True)
        st.markdown("<hr><br>", unsafe_allow_html=True)
        with st.expander("Dataset PTA Trunojoyo"):
            st.write(data)
        with st.expander("Data Per Baris"):
            data_rows = list(data.index)
            selected_row = st.selectbox("Pilih Baris Data:", data_rows)
            st.write(data.loc[selected_row])
        with st.expander("Penjelasan Sistem"):
            st.markdown(
                "<h3 style='text-align:justify'>Halaman Menu Sidebar Sistem</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h5 style='text-align:justify'>1. Data : Menampilkan Data PTA</h5>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h5 style='text-align:justify'>2. Pre Processing Data : Mengolah dan Menampilkan data yang sudah di pre processing </h5>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h5 style='text-align:justify'>3. Ektraksi Fitur : Menampilkan hasil Jumlah Term,One Hot Encoding,TF-IDF,dan Log Term</h5>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h5 style='text-align:justify'>4. Skip Gram : Menampilkan hasil Skip Gram</h5>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h5 style='text-align:justify'>5. Topic Modeling : Menampilkan hasil dari topic modeling menggunakan LDA</h5>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h5 style='text-align:justify'>6. Klasifikasi Data : Menampilkan hasil dari Klasifikasi menggunakan Naive Bayes,KNN,Decision Tree</h5>",
                unsafe_allow_html=True,
            )
        
    #pre processing data
    elif choice == "Pre processing data":
        st.title("Pengolahan Data PTA Trunojoyo")
        st.markdown("<h4>Pre processing data</h4>", unsafe_allow_html=True)
        st.markdown("<hr><br>", unsafe_allow_html=True)
        # Cek missing value
        with st.expander("Cek Data Kosong"):
            st.write(data.isnull().sum())
            if st.button("Hapus Data Kosong"):
                data.dropna(inplace=True)
                st.write("Data kosong telah dihapus. Jumlah Data Kosong sekarang:")
                st.write(data.isnull().sum())

        # Cleaning Data
        with st.expander("Cleaning Data"):
            st.markdown("<h6>Proses Pembersihan Teks (Abstrak) yang meliputi :</h6>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h7>- Tag HTML Bawaan</h7>", unsafe_allow_html=True)
                st.markdown("<h7>- LowerCase Data</h7>", unsafe_allow_html=True)
                st.markdown("<h7>- Spasi pada teks</h7>", unsafe_allow_html=True)
            with col2:
                st.markdown("<h7>- Tanda baca dan karakter spesial</h7>", unsafe_allow_html=True)
                st.markdown("<h7>- Nomor</h7>", unsafe_allow_html=True)
                st.markdown("<h7>- Komponen Lainnya</h7>", unsafe_allow_html=True)
            if st.button("Cleaning Data"):
                data['Abstrak'] = data['Abstrak'].apply(lambda x: cleaning(x))
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<h6>Data Cleaning Abstrak :</h6>",unsafe_allow_html=True)
                data.dropna(inplace=True)
                st.write(data['Abstrak'])

        # Tokenisasi Data
        with st.expander("Tokenisasi"):
            st.markdown("<h6>Proses Memisahkan sebuah Dokumen menjadi susunan per kata / term</h6>", unsafe_allow_html=True)
            if st.button("Tokenisasi Data"):
                data.dropna(inplace=True)
                data['Abstrak'] = data['Abstrak'].apply(lambda x: cleaning(x))
                data['Abstrak'] = data['Abstrak'].fillna('')
                data['abstrak_tokens'] = data['Abstrak'].apply(lambda x: tokenize_text(x))
                st.write(data[['Abstrak','abstrak_tokens']])

        # Stopword Data
        with st.expander("StopWord"):
            st.markdown("<h6>Mengubah isi dokumen sesuai dengan kamus data</h6>", unsafe_allow_html=True)
            if st.button("StopWord Data"):
                data['Abstrak'] = data['Abstrak'].apply(lambda x: cleaning(x))
                data['Abstrak'] = data['Abstrak'].fillna('')
                data['abstrak_tokens'] = data['Abstrak'].apply(lambda x: tokenize_text(x))
                data['abstrak_tokens'] = data['abstrak_tokens'].apply(lambda x: remove_stopwords(x))
                data['Abstrak'] = data['abstrak_tokens'].apply(lambda tokens: ' '.join(tokens))
                st.write(data[['Abstrak','abstrak_tokens']])

        # Steaming Data
        with st.expander("Steaming"):
            st.markdown("<h6>Mengubah kata menjadi bentuk dasar</h6>", unsafe_allow_html=True)
            if st.button("Steaming Data"):
                data = pd.read_csv('https://raw.githubusercontent.com/davata1/Dataset/main/DataSteamingg.csv')
                st.write(data[['Abstrak','abstrak_tokens']])
    # Ekstraksi Fitur 
    elif choice == "Ekstraksi Fitur":
        st.title("Pengolahan Data PTA Trunojoyo")
        st.markdown("<h4>Ekstraksi Fitur</h4>", unsafe_allow_html=True)
        st.markdown("<hr><br>", unsafe_allow_html=True)
        data = pd.read_csv('https://raw.githubusercontent.com/davata1/Dataset/main/DataSteamingg.csv')
        # Term Frekuensi
        with st.expander("Term Frekuensi"):
            with open("model/count_vectorizer_model.pkl", "rb") as file:
                count_vectorizer = pickle.load(file)
            st.markdown("<h6>Seberapa sering sebuah kata atau term tertentu muncul dalam sebuah dokumen </h6>", unsafe_allow_html=True)
            X_count = count_vectorizer.transform(np.array(data['Abstrak'].values.astype('U')))
            terms_count = count_vectorizer.get_feature_names_out()
            data_countvect = pd.DataFrame(data=X_count.toarray(), columns=terms_count)
            data_countvect['label'] = data['Label'].values
            st.write(data_countvect)
        
        #One Hot Encoding
        with st.expander("One Hot Encoding"):
            st.markdown("<h6>mengubah data kategori atau data yang memiliki kelas diskrit menjadi bentuk biner</h6>", unsafe_allow_html=True)
            with open("model/df_binary_model.pkl", "rb") as file:
                data_binary = pickle.load(file)
            st.write(data_binary)
        
        #TF-IDF
        with st.expander("TF-IDF"):
            st.markdown("<h6>memberikan skor pada sebuah kata (term) berdasarkan seberapa penting kata tersebut dalam suatu dokumen dalam kumpulan dokumen</h6>", unsafe_allow_html=True)
            with open("model/tfidf_vectorizer_model.pkl", "rb") as file:
                vectorizer = pickle.load(file)
            X_tfidf = vectorizer.transform(data['Abstrak'].values.astype('U').tolist())
            terms = vectorizer.get_feature_names_out()
            data_tfidfvect = pd.DataFrame(data=X_tfidf.toarray(), columns=terms)
            st.write(data_tfidfvect)

        #Log Frekuensi
        with st.expander("Log Frekuensi"):
            st.markdown("<h6>Mengubah frekuensi kata atau term dalam teks menjadi skala logaritmik.</h6>", unsafe_allow_html=True)
            with open("model/df_log_model.pkl", "rb") as file:
                data_log = pickle.load(file)
            st.write(data_log)

    elif choice == "Skip Gram":
        st.title("Pengolahan Data PTA Trunojoyo")
        st.markdown("<h4>Skip Gram</h4>", unsafe_allow_html=True)
        st.markdown("<hr><br>", unsafe_allow_html=True)
        model_word2vec = Word2Vec.load("model/word2vec_model")
        word = st.text_input("Masukkan kata untuk mencari kemiripan:")
        similar_words = []
        if st.button("Cari Kemiripan"):
            if word in model_word2vec.wv:
                similar_words = model_word2vec.wv.most_similar(word)
                st.write(f"Kata yang dekat dengan '{word}':")
            else:
                st.write(f"Kata '{word}' tidak ditemukan dalam model Word2Vec.")
        for w, sim in similar_words:
            st.write(f"{w}: {sim:.4f}")

    elif choice == "Topic Modeling":
        st.title("Pengolahan Data PTA Trunojoyo")
        st.markdown("<h4>Topic Modeling</h4>", unsafe_allow_html=True)
        st.markdown("<hr><br>", unsafe_allow_html=True)
        data = pd.read_csv('https://raw.githubusercontent.com/davata1/Dataset/main/DataSteamingg.csv')
        data_tfidf = pd.read_csv('https://raw.githubusercontent.com/davata1/Dataset/main/Data_TF-IDF.csv')
        # with open("model/lda_model.pkl", "rb") as file:
        with open("model/lda_model100.pkl", "rb") as file:
            lda_model = pickle.load(file)
        with st.expander("Update Bobot"):
            with open("model/tfidf_vectorizer_model.pkl", "rb") as file:
                vectorizer = pickle.load(file)
            terms = vectorizer.get_feature_names_out()
            X_tfidf = vectorizer.fit_transform(data['Abstrak'].values.astype('U'))
            w1 = lda_model.transform(X_tfidf)
            h1 = lda_model.components_
            st.write(w1)    
        with st.expander("Proporsi topic pada dokumen"):            
            n_components = 100
            colnames = ["Topic" + str(i) for i in range(n_components)]
            docnames = ["Doc" + str(i) for i in range(len(data['Abstrak']))]
            df_doc_topic = pd.DataFrame(np.round(w1, 2), columns=colnames, index=docnames)
            df_doc_topic['label'] = data['Label'].values
            st.write(df_doc_topic)
        with st.expander("Proporsi topic pada Kata"):
            label = []
            for i in range(1, (lda_model.components_.shape[1] + 1)):
                masukan = data_tfidf.columns[i - 1]
                label.append(masukan)
            data_topic_word = pd.DataFrame(lda_model.components_, columns=label)
            data_topic_word.rename(index={0: "Topik 1", 1: "Topik 2", 2: "Topik 3"}, inplace=True)
            st.write(data_topic_word.transpose())

    elif choice == "Klasifikasi Data":
        st.title("Pengolahan Data PTA Trunojoyo")
        st.markdown("<h4>Klasifikasi Data</h4>", unsafe_allow_html=True)
        st.markdown("<hr><br>", unsafe_allow_html=True)
        data = pd.read_csv('https://raw.githubusercontent.com/davata1/Dataset/main/DataSteamingg.csv')
        new_data = []
        new_text = st.text_input("Masukkan teks baru untuk diprediksi:", "")
        if new_text:
            new_data.append(new_text)
            df_baru = pd.DataFrame({'Abstrak': new_data})
            df_baru['Abstrak'] = df_baru['Abstrak'].apply(lambda x: cleaning(x))
            df_baru['Abstrak'] = df_baru['Abstrak'].fillna('')
            df_baru['abstrak_tokens'] = df_baru['Abstrak'].apply(lambda x: tokenize_text(x))
            df_baru['abstrak_tokens'] = df_baru['abstrak_tokens'].apply(lambda x: remove_stopwords(x))
            df_baru['abstrak_tokens'] = df_baru['abstrak_tokens'].apply(lambda x: stem_text(' '.join(x)).split(' '))
            df_baru['Abstrak'] = df_baru['abstrak_tokens'].apply(lambda tokens: ' '.join(tokens))
            st.markdown("<h4>Hasil Processing</h4>", unsafe_allow_html=True)
            st.write(df_baru[['Abstrak','abstrak_tokens']])
            st.markdown("<h4>Pilih model Klasifikasi Data</h4>", unsafe_allow_html=True)
            data_count = pd.read_csv('https://raw.githubusercontent.com/davata1/Dataset/main/Data_CountVectorize.csv')
            data_count['labels'] = data['Label'].values
            
            # Count Data
            with open("model/count_vectorizer_model.pkl", "rb") as file:
                    count_vectorizer = pickle.load(file)
            X_count = count_vectorizer.fit_transform(data['Abstrak'].values.astype('U'))

            # Count Data Baru
            X_count_baru = count_vectorizer.transform(df_baru['Abstrak'])

            # Topic Modeling Data
            # with open("model/lda_model.pkl", "rb") as file:
            with open("model/lda_model100.pkl", "rb") as file:
                lda_model = pickle.load(file)
            w1 = lda_model.transform(X_count)
            h1 = lda_model.components_
            df_doc_topic = pd.DataFrame(np.round(w1,2))
            df_doc_topic['label'] = data['Label'].values
            
            # Topic Modeling Data
            w1_baru = lda_model.transform(X_count_baru)

            # Training dengan topic modeling
            X = df_doc_topic.drop('label', axis=1)
            y = df_doc_topic['label']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            #Training dengan tf-idf
            X_2 = data_count.drop('labels', axis=1)
            y_2 = data_count['labels']
            X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=42)

            with st.expander("Naive Bayes"):
                with open("model/naive_bayes_model.pkl", "rb") as file:
                    naive_bayes_classifier = pickle.load(file)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("<h4>Dengan Data Topic Modeling</h4>", unsafe_allow_html=True)
                        naive_bayes_classifier.fit(X_train, y_train)
                        y_pred = naive_bayes_classifier.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        y_pred_baru = naive_bayes_classifier.predict(w1_baru)
                        st.markdown("<h4>Hasil Klasifikasi</h4>", unsafe_allow_html=True)
                        st.write(y_pred_baru)
                        st.write(accuracy)
                        st.write(classification_report(y_test, y_pred))
                    with col2:
                        st.markdown("<h4>Dengan Data Asli</h4>", unsafe_allow_html=True)
                        naive_bayes_classifier.fit(X_train_2, y_train_2)
                        y_pred2 = naive_bayes_classifier.predict(X_test_2)
                        accuracy = accuracy_score(y_test_2, y_pred2)
                        y_pred_baru2 = naive_bayes_classifier.predict(X_count_baru)
                        st.markdown("<h4>Hasil Klasifikasi</h4>", unsafe_allow_html=True)
                        st.write(y_pred_baru2)
                        st.write(accuracy)
                        st.write(classification_report(y_test_2, y_pred2))
            with st.expander("KNN"):
                with open("model/knn_model.pkl", "rb") as file:
                    neigh = pickle.load(file)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<h4>Dengan Data Topic Modeling</h4>", unsafe_allow_html=True)
                    knn = neigh.fit(X_train, y_train)
                    y_pred_knn = knn.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred_knn)
                    y_pred_knn_baru = knn.predict(w1_baru)
                    st.markdown("<h4>Hasil Klasifikasi</h4>", unsafe_allow_html=True)
                    st.write(y_pred_knn_baru)
                    st.write(accuracy)
                    st.write(classification_report(y_test, y_pred_knn))
                with col2:
                    st.markdown("<h4>Dengan Data Asli</h4>", unsafe_allow_html=True)
                    knn_2 = neigh.fit(X_train_2, y_train_2)
                    y_pred_knn_2 = knn_2.predict(X_test_2)
                    accuracy = accuracy_score(y_test_2, y_pred_knn_2)
                    y_pred_knn_baru2 = knn_2.predict(X_count_baru)
                    st.markdown("<h4>Hasil Klasifikasi</h4>", unsafe_allow_html=True)
                    st.write(y_pred_knn_baru2)
                    st.write(accuracy)
                    st.write(classification_report(y_test_2, y_pred_knn_2))
            with st.expander("Decision Tree"):
                with open("model/tree_model.pkl", "rb") as file:
                    clf = pickle.load(file)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<h4>Dengan Data Topic Modeling</h4>", unsafe_allow_html=True)
                    decision_tree = clf.fit(X_train, y_train)
                    y_pred_clf = decision_tree.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred_clf)
                    y_pred_clf_baru = decision_tree.predict(w1_baru)
                    st.markdown("<h4>Hasil Klasifikasi</h4>", unsafe_allow_html=True)
                    st.write(y_pred_clf_baru)
                    st.write(accuracy)
                    st.write(classification_report(y_test, y_pred_clf))
                with col2:
                    st.markdown("<h4>Dengan Data Asli</h4>", unsafe_allow_html=True)
                    decision_tree_2 = clf.fit(X_train_2, y_train_2)
                    y_pred_clf_2 = decision_tree_2.predict(X_test_2)
                    accuracy = accuracy_score(y_test_2, y_pred_clf_2)
                    y_pred_clf_baru2 = decision_tree_2.predict(X_count_baru)
                    st.markdown("<h4>Hasil Klasifikasi</h4>", unsafe_allow_html=True)
                    st.write(y_pred_clf_baru2)
                    st.write(accuracy)
                    st.write(classification_report(y_test_2, y_pred_clf_2))
if __name__ == '__main__':
    main()