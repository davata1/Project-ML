import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('/content/drive/MyDrive/kuliah/data/Produksi Tanaman Cabe.csv')

# Konversi data Tahun menjadi numerik
df['Tahun'] = pd.to_numeric(df['Tahun'])

# Streamlit app
st.title("Aplikasi Prediksi Produksi Cabe")

# Tampilkan dataset
st.subheader("Dataset Produksi Cabe")
st.dataframe(df)

# Tampilkan grafik produksi per provinsi
st.subheader("Grafik Produksi Cabe per Provinsi")
plt.figure(figsize=(10, 6))
sns.barplot(x='Provinsi', y='Produksi', data=df)
plt.xticks(rotation=90)
plt.xlabel('Provinsi')
plt.ylabel('Produksi')
plt.title('Produksi Cabe per Provinsi dari Tahun 2003-2023')
st.pyplot(plt)

# Pilih provinsi untuk prediksi
provinsi = df['Provinsi'].unique()
selected_provinsi = st.selectbox("Pilih Provinsi", provinsi)

# Input 3 tahun produksi terakhir
tahun_terakhir_1 = st.number_input("Produksi Tahun Terakhir 1", min_value=0)
tahun_terakhir_2 = st.number_input("Produksi Tahun Terakhir 2", min_value=0)
tahun_terakhir_3 = st.number_input("Produksi Tahun Terakhir 3", min_value=0)

if st.button("Prediksi Produksi Tahun Berikutnya"):
    # Siapkan data untuk model
    X = pd.DataFrame({'Tahun': [2021, 2022, 2023]})
    y = pd.Series([tahun_terakhir_1, tahun_terakhir_2, tahun_terakhir_3])

    # Normalisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data menjadi training dan testing (meskipun kita hanya menggunakan data ini)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Buat model linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediksi untuk tahun berikutnya
    tahun_prediksi = 2024
    X_prediksi = scaler.transform(pd.DataFrame({'Tahun': [tahun_prediksi]}))
    y_prediksi = model.predict(X_prediksi)

    st.subheader(f"Hasil Prediksi Produksi Cabe untuk Provinsi {selected_provinsi} di Tahun {tahun_prediksi}:")
    st.write(f'Produksi: {y_prediksi[0]:.2f}')

# Evaluasi model (hanya untuk informasi)
if st.button("Evaluasi Model"):
    all_y_test = [tahun_terakhir_1, tahun_terakhir_2, tahun_terakhir_3]
    all_y_pred = [y_prediksi[0]]  # Hanya satu prediksi

    mse = mean_squared_error(all_y_test, all_y_pred)
    rmse = mean_squared_error(all_y_test, all_y_pred, squared=False)
    r_squared = r2_score(all_y_test, all_y_pred)

    st.subheader("Evaluasi Model")
    st.write(f'MSE: {mse:.2f}')  
    st.write(f'RMSE: {rmse:.2f}')  
    st.write(f'R-squared: {r_squared:.2f}')

# Menampilkan total produksi cabai per provinsi
total_produksi = df.groupby('Provinsi')['Produksi'].sum()
st.subheader('Produksi cabai per Provinsi dari Tahun 2003-2023:')
st.write(total_produksi)