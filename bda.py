import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import numpy as np
import pickle
from PIL import Image
import requests
import io

# Load dataset
df = pd.read_csv('https://github.com/davata1/Project-ML/raw/refs/heads/main/Produksi%20Tanaman%20Cabe.csv')

# Streamlit app
st.title("Aplikasi Prediksi Produksi Cabe")

# Kategori dengan tabs
kategori = st.tabs(["Prediksi", "Klasifikasi"])

# Prediction Tab
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

# Classification Tab
with kategori[1]:
    # Load the KNN model
    # model_url = 'https://github.com/davata1/Project-ML/raw/main/knn_model.pkl'
    # response = requests.get(model_url)
    # model = pickle.loads(response.content)  # Load the model from the URL
    model = load_model ('knn_model.pkl')

    classes = ["Sehat", "Hama", "Bercak"]
    
    # Menu pilihan
    menu = st.selectbox("Capture Option:", ["Upload Photo", "Camera"])

    if menu == "Upload Photo":
        uploaded_file = st.file_uploader("Select photo", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Photo', use_column_width=True)
            
            # Mengubah gambar menjadi bentuk yang sesuai untuk prediksi
            resized_image = image.resize((128, 128))  # Resize to match model input
            processed_image = np.array(resized_image) / 255.0  # Normalize the image
            input_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

            # Melakukan prediksi menggunakan model
            prediction = model.predict(input_image)
            class_index = np.argmax(prediction[0])
            class_name = classes[class_index]

            # Menampilkan hasil prediksi
            st.success(f"Hasil Prediksi: {class_name}")