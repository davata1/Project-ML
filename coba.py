import pandas as pd
import requests
from io import BytesIO

# URL file Excel di GitHub
url = 'https://github.com/davata1/Project-ML/raw/refs/heads/main/sinjaymadura.xlsx'

# Mengunduh file dari URL
response = requests.get(url)

# Cek apakah unduhan berhasil
if response.status_code == 200:
    # Membaca file Excel dari response content
    data = pd.read_excel(BytesIO(response.content))

    # Cek apakah DataFrame tidak kosong
    if not data.empty:
        st.write("DataFrame shape:", data.shape)
        st.write("DataFrame columns:", data.columns.tolist())
        st.write(f"Sample label powerset: {data.iloc[:5, 1].tolist()}")
    else:
        st.error("DataFrame kosong. Pastikan data berhasil dimuat.")
else:
    st.error("Gagal mengunduh file. Periksa URL atau koneksi.")
