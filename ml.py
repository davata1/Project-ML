import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Judul Aplikasi
st.title("PENAMBANGAN DATA")
st.write("##### Nama  : Dhafa Febriyan Wiranata")
st.write("##### Nim   : 200411100169")
st.write("##### Kelas : Penambangan Data B")

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
description = st.sidebar.button("Description")
upload_data = st.sidebar.button("Upload Data")
preprocessing = st.sidebar.button("Preprocessing")


# Deskripsi
if description:
    st.write("###### Data Set : Human Stress Detection in and through Sleep - Deteksi Stres Manusia di dalam dan melalui Tidur")
    st.write("###### Sumber Data Set dari Kaggle : https://www.kaggle.com/datasets/laavanya/human-stress-detection-in-and-through-sleep?select=SaYoPillow.csv")

# Upload Data
if upload_data:
    st.write("###### DATASET YANG DIGUNAKAN")
    df = pd.read_csv('https://raw.githubusercontent.com/davata1/Project-Pendat/main/SaYoPillow.csv')
    st.dataframe(df)

# Preprocessing
if preprocessing:
    st.subheader("Normalisasi Data")
    st.write("Rumus Normalisasi Data :")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)

    # Mendefinisikan Variabel X dan Y
    X = df.drop(columns=['sl'])
    y = df['sl'].values

    # Normalisasi Nilai X
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.sl).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '1': [dumies[0]],
        '2': [dumies[1]],
        '3': [dumies[2]],
        '4': [dumies[3]],
        '5': [dumies[4]]
    })
    st.write(labels)
