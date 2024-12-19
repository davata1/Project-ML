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
    
    # Clear the current figure
    plt.clf()
    
    # Plot data produksi per provinsi
    plt.figure(figsize=(12, 6))
    for prov in df['Provinsi'].unique():
        province_data = df[df['Provinsi'] == prov]
        plt.plot(province_data['Tahun'], province_data['Produksi'], marker='o', markersize=5, label=prov)  # Adjust markersize if needed

    plt.xlabel('Tahun')
    plt.ylabel('Produksi Cabe')
    plt.title('Perbandingan Produksi Cabe per Daerah')
    plt.legend(loc='upper left')
    
    # Render the plot
    st.pyplot(plt)




