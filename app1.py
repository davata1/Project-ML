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

# Convert 'Tahun' to numeric
df['Tahun'] = pd.to_numeric(df['Tahun'])

# Streamlit app
st.title("Aplikasi Prediksi Produksi Cabe")

# Kategori dengan tabs
kategori = st.tabs(["Dataset", "Grafik", "Prediksi", "Evaluasi"])

with kategori[0]:
    st.subheader("Dataset Produksi Cabe")
    st.dataframe(df)

with kategori[1]:
    st.subheader("Grafik Produksi Cabe per Provinsi")
    
    # Plot data produksi per provinsi
    plt.figure(figsize=(12, 6))
    for prov in df['Provinsi'].unique():
        province_data = df[df['Provinsi'] == prov]
        plt.plot(province_data['Tahun'], province_data['Produksi'], marker='o', label=prov)

    plt.xlabel('Tahun')
    plt.ylabel('Produksi Cabe')
    plt.title('Perbandingan Produksi Cabe per Daerah')
    plt.legend()
    st.pyplot(plt)

    # Total produksi per tahun
    plt.figure(figsize=(10, 6))
    df.groupby('Tahun')['Produksi'].sum().plot()
    plt.xlabel('Tahun')
    plt.ylabel('Produksi')
    plt.title('Total Produksi Cabe dari Tahun 2003-2023')
    st.pyplot(plt)

    # Visualisasi Produksi per Provinsi
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Provinsi', y='Produksi', data=df)
    plt.xlabel('Provinsi')
    plt.xticks(rotation=90)
    plt.ylabel('Produksi')
    plt.title('Produksi Cabai per Provinsi dari Tahun 2003-2023')
    st.pyplot(plt)

with kategori[2]:
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

with kategori[3]:
    st.subheader("Evaluasi")
                     # Inisialisasi untuk evaluasi
    all_y_test = []
    all_y_pred = []

    # Model untuk setiap provinsi
    for prov in provinsi:
        province_data = df[df['Provinsi'] == prov]
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

    # Evaluasi model
    mse = mean_squared_error(all_y_test, all_y_pred)  # Hitung MSE
    rmse = np.sqrt(mse)  # Calculate RMSE from MSE
    r_squared = r2_score(all_y_test, all_y_pred)  # Hitung R-squared

    st.write(f'MSE: {mse:.2f}')  
    st.write(f'RMSE: {rmse:.2f}')  
    st.write(f'R-squared: {r_squared:.2f}')


# Menampilkan hasil prediksi untuk tahun 2024, 2025, dan 2026
st.subheader("Hasil Prediksi Produksi Cabe untuk Tahun 2024, 2025, dan 2026:")
for prov in provinsi:
    if prov in prediksi:
        for tahun in prediksi[prov]:
            st.write(f'{prov} - Tahun {tahun}: {prediksi[prov][tahun]:.2f}')
