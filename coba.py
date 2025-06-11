# Memuat data dari file Excel
data = pd.read_excel('https://github.com/davata1/Project-ML/blob/main/sinjaymadura.xlsx')

# Cek apakah DataFrame tidak kosong
if not data.empty:
    st.write("DataFrame shape:", data.shape)
    st.write("DataFrame columns:", data.columns.tolist())
    st.write(f"Sample label powerset: {data.iloc[:5, 1].tolist()}")
else:
    st.error("DataFrame kosong. Pastikan data berhasil dimuat.")

# Lanjutkan dengan proses lainnya...
