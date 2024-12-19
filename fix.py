import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pickle
from PIL import Image

# Load dataset
df = pd.read_csv('https://github.com/davata1/Project-ML/raw/refs/heads/main/Produksi%20Tanaman%20Cabe.csv')

# Streamlit app
st.title("Aplikasi Prediksi Produksi Cabe")

# Kategori dengan tabs
kategori = st.tabs(["Prediksi", "Klasifikasi"])

# Prediction Tab
with kategori[0]:
    st.subheader("Grafik Produksi Cabe per Provinsi")
    
    # Plot data produksi per provinsi
    plt.figure(figsize=(12, 6))
    for prov in df['Provinsi'].unique():
        province_data = df[df['Provinsi'] == prov]
        plt.plot(province_data['Tahun'], province_data['Produksi'], marker='o', label=prov)

        # Menambahkan prediksi untuk tahun 2024, 2025, dan 2026
        for tahun in [2024, 2025, 2026]:
            if (prov, tahun) in prediksi:  # Check if prediction exists
                plt.plot(tahun, prediksi[(prov, tahun)], 'ro', markersize=8)  # Titik merah untuk prediksi

    plt.xlabel('Tahun')
    plt.ylabel('Produksi Cabe')
    plt.title('Perbandingan Produksi Cabe per Daerah')
    plt.legend(loc='upper left')  # Optional: specify legend location
    st.pyplot(plt)

# Classification Tab
with kategori[1]:
    # Load the KNN model
    model_url = 'https://github.com/davata1/Project-ML/raw/main/knn_model.pkl'  # Use the raw link
    model = pickle.load(open('knn_model.pkl', 'rb'))  # Load the model from a local file

    classes = ["_BrownSpot", "_Hispa", "_LeafBlast", "_Healthy"]
    
    # Menu pilihan
    menu = st.selectbox("Capture Option:", ["Upload Photo", "Camera"])

    if menu == "Upload Photo":
        uploaded_file = st.file_uploader("Select photo", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Photo', use_column_width=True)
            # Mengubah gambar menjadi bentuk yang sesuai untuk prediksi
            resized_image = image.resize((128, 128))
                        # Normalize the image
            processed_image = np.array(resized_image) / 255.0
            input_image = np.expand_dims(processed_image, axis=0)

            # Melakukan prediksi menggunakan model
            prediction = model.predict(input_image)
            class_index = np.argmax(prediction[0])
            class_name = classes[class_index]

            # Menampilkan hasil prediksi
            st.success(f"Hasil Prediksi: {class_name}")

# Optional: Add a section to display the model's performance metrics if available
# You can load the metrics from a file or calculate them based on a test dataset
