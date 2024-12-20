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
kategori = st.tabs(["Prediksi", "Klasifikasi","grafik"])

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
with kategori[2]:
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

    # Menampilkan total produksi cabai per provinsi
    total_produksi = df.groupby('Provinsi')['Produksi'].sum()
    st.subheader('Produksi cabai per Provinsi dari Tahun 2003-2023:')
    st.write(total_produksi)

# Optional: Add a section to display the model's performance metrics if available
# You can load the metrics from a file or calculate them based on a test dataset