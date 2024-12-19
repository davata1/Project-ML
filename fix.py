import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load dataset
df = pd.read_csv('https://github.com/davata1/Project-ML/raw/refs/heads/main/Produksi%20Tanaman%20Cabe.csv')

# Streamlit app
st.title("Pendekatan Big Data Analisis Dalam Sektor Pertanian")

# Kategori dengan tabs
kategori = st.tabs(["Prediksi"])

prediksi = {}
all_y_test = []
all_y_pred = []

# Prediction Tab
with kategori[0]:
    st.subheader("Grafik Produksi Cabe per Provinsi")
    
    # Daftar tahun yang ingin diprediksi
    tahun_prediksi_list = [2024, 2025, 2026]

    # Create a figure for the plot
    plt.figure(figsize=(12, 6))

    # Loop through each province to train the model and make predictions
    for prov in df['Provinsi'].unique():
        # Ambil data untuk provinsi tertentu
        province_data = df[df['Provinsi'] == prov]

        # Siapkan data untuk model
        X = province_data[['Tahun']]
        y = province_data['Produksi']

        # Normalisasi data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data menjadi training dan testing
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Buat model linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Prediksi pada data test
        y_pred = model.predict(X_test)

        # Simpan hasil prediksi dan nilai sebenarnya untuk RMSE keseluruhan
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

        # Prediksi produksi cabe untuk tahun 2024, 2025, dan 2026
        for tahun_prediksi in tahun_prediksi_list:
            X_prediksi = scaler.transform(pd.DataFrame({'Tahun': [tahun_prediksi]}))
            y_prediksi = model.predict(X_prediksi)
            prediksi[(prov, tahun_prediksi)] = y_prediksi[0]

        # Plot data produksi per provinsi
        plt.plot(province_data['Tahun'], province_data['Produksi'], marker='o', label=prov)

    # Menambahkan prediksi untuk tahun 2024, 2025, dan 2026
    for prov in df['Provinsi'].unique():
        for tahun in tahun_prediksi_list:
            if (prov, tahun) in prediksi:  # Check if the prediction exists
                plt.plot([tahun], [prediksi[(prov, tahun)]], 'ro')  # Titik merah untuk prediksi

    plt.xlabel('Tahun')
    plt.ylabel('Produksi Cabe')
    plt.title('Perbandingan Produksi Cabe per Daerah dan Prediksi 2024-2026')
    plt.legend()
    st.pyplot(plt)
    tahun_prediksi_list = [2024, 2025, 2026]  # List tahun prediksi
    for prov in df['Provinsi'].unique():
        for tahun in tahun_prediksi_list:
            plt.plot([tahun], [prediksi[(prov, tahun)]], 'ro')  # Titik merah untuk prediksi

 

