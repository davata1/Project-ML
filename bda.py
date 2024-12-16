import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import cv2

# Load dataset
df = pd.read_csv('https://github.com/davata1/Project-ML/raw/refs/heads/main/Produksi%20Tanaman%20Cabe.csv')

# Streamlit app
st.title("Aplikasi Prediksi Produksi Cabe")

# Kategori dengan tabs
kategori = st.tabs(["Prediksi", "Klasifikasi"])

with kategori[0]:
    st.subheader("Prediksi Produksi Cabe")
    provinsi = df['Provinsi'].unique()
    selected_provinsi = st.selectbox("Pilih Provinsi", provinsi)

    # Input 20 tahun produksi dari 2003 hingga 2023
    tahun_produksi = {}
    for year in range(2003, 2024):
        tahun_produksi[year] = st.number_input(f"Produksi Tahun {year}", min_value=0)

    # Inisialisasi dictionary untuk menyimpan prediksi
    prediksi = {}

    if st.button("Prediksi Produksi Tahun Berikutnya"):
        # Siapkan data untuk model
        X = pd.DataFrame({'Tahun': list(tahun_produksi.keys())})
        y = pd.Series(list(tahun_produksi.values()))

        # Normalisasi data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data menjadi training dan testing
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Buat model linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Prediksi untuk tahun berikutnya (2024, 2025, 2026)
        tahun_prediksi = [2024, 2025, 2026]
        prediksi[selected_provinsi] = {}

        for tahun in tahun_prediksi:
            X_prediksi = scaler.transform(pd.DataFrame({'Tahun': [tahun]}))
            y_prediksi = model.predict(X_prediksi)
            prediksi[selected_provinsi][tahun] = y_prediksi[0]

        st.subheader(f"Hasil Prediksi Produksi Cabe untuk Provinsi {selected_provinsi}:")
        for tahun in tahun_prediksi:
            st.write(f'Tahun {tahun}: Produksi: {prediksi[selected_provinsi][tahun]:.2f}')

with kategori[1]:
    st.subheader("Klasifikasi Gambar Produksi Cabe")
    
    # Upload image
    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image, caption='Gambar yang Diunggah', use_column_width=True)

        # Load KNN model
        knn_model = joblib.load('knn.pkl')

        # Ekstraksi fitur dari gambar
        def extract_hsv_features(image):
            hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv_img], [0], None, [256], [0, 256])
            hist_s = cv2.calcHist([hsv_img], [1], None, [256], [0, 256])
            hist_v = cv2.calcHist([hsv_img], [2], None, [256], [0, 256])
            hist_h = hist_h / hist_h.sum()
            hist_s = hist_s / hist_s.sum()
            hist_v = hist_v / hist_v.sum()
            features = np.concatenate((hist_h.flatten(), hist_s.flatten(), hist_v.flatten()))
            return features

        # Extract features from the uploaded image
        features = extract_hsv_features(image).reshape(1, -1)

        # Make prediction
        predicted_class = knn_model.predict(features)

        # Display predicted class
        st.subheader("Hasil Prediksi Klasifikasi Gambar:")
        st.write(f'Kelas yang Diprediksi: {predicted_class[0]}')
