# Script untuk debug masalah loading Excel dari GitHub
import pandas as pd
import streamlit as st
import requests
from io import BytesIO

# Method 1: Direct pandas read_excel (original method)
st.write("=== Method 1: Direct pandas read_excel ===")
try:
    data = pd.read_excel('https://github.com/davata1/Project-ML/raw/refs/heads/main/sinjaymadura.xlsx')
    st.write("✅ Berhasil memuat dengan pandas direct")
    st.write("DataFrame shape:", data.shape)
    st.write("DataFrame columns:", data.columns.tolist())
    st.write("Sample data:")
    st.dataframe(data.head())
except Exception as e:
    st.write("❌ Error dengan pandas direct:")
    st.write(str(e))

# Method 2: Using requests first, then pandas
st.write("\n=== Method 2: Using requests + BytesIO ===")
try:
    url = 'https://github.com/davata1/Project-ML/raw/refs/heads/main/sinjaymadura.xlsx'
    response = requests.get(url)
    response.raise_for_status()  # Akan raise exception jika status code bukan 200
    
    st.write(f"✅ Response status: {response.status_code}")
    st.write(f"Content length: {len(response.content)} bytes")
    st.write(f"Content type: {response.headers.get('content-type', 'Unknown')}")
    
    # Load into pandas
    data = pd.read_excel(BytesIO(response.content))
    st.write("✅ Berhasil memuat dengan requests + BytesIO")
    st.write("DataFrame shape:", data.shape)
    st.write("DataFrame columns:", data.columns.tolist())
    st.write("Sample data:")
    st.dataframe(data.head())
    
except requests.exceptions.RequestException as e:
    st.write("❌ Error dengan requests:")
    st.write(str(e))
except Exception as e:
    st.write("❌ Error dengan pandas:")
    st.write(str(e))

# Method 3: Check if URL is accessible
st.write("\n=== Method 3: URL Accessibility Check ===")
try:
    url = 'https://github.com/davata1/Project-ML/raw/refs/heads/main/sinjaymadura.xlsx'
    response = requests.head(url)  # HEAD request untuk cek accessibility
    st.write(f"URL Status Code: {response.status_code}")
    st.write(f"Headers: {dict(response.headers)}")
    
    if response.status_code == 200:
        st.write("✅ URL dapat diakses")
    else:
        st.write("❌ URL tidak dapat diakses dengan benar")
        
except Exception as e:
    st.write("❌ Error checking URL:")
    st.write(str(e))

# Method 4: Alternative - Download and save locally first
st.write("\n=== Method 4: Download and Save Locally ===")
try:
    import os
    url = 'https://github.com/davata1/Project-ML/raw/refs/heads/main/sinjaymadura.xlsx'
    
    # Download file
    response = requests.get(url)
    response.raise_for_status()
    
    # Save temporarily
    local_filename = 'temp_sinjaymadura.xlsx'
    with open(local_filename, 'wb') as f:
        f.write(response.content)
    
    st.write(f"✅ File downloaded, size: {os.path.getsize(local_filename)} bytes")
    
    # Load with pandas
    data = pd.read_excel(local_filename)
    st.write("✅ Berhasil memuat dari file lokal")
    st.write("DataFrame shape:", data.shape)
    st.write("DataFrame columns:", data.columns.tolist())
    st.write("Sample data:")
    st.dataframe(data.head())
    
    # Clean up
    os.remove(local_filename)
    st.write("✅ File temporary dihapus")
    
except Exception as e:
    st.write("❌ Error dengan download local:")
    st.write(str(e))

# Additional debugging information
st.write("\n=== Additional Debug Info ===")
st.write(f"Pandas version: {pd.__version__}")
st.write(f"Python version: {sys.version}")

# Test with your original logic
st.write("\n=== Testing Original Logic ===")
try:
    data = pd.read_excel('https://github.com/davata1/Project-ML/raw/refs/heads/main/sinjaymadura.xlsx')
    
    # Cek apakah DataFrame tidak kosong
    if not data.empty:
        st.write("✅ DataFrame tidak kosong")
        st.write("DataFrame shape:", data.shape)
        st.write("DataFrame columns:", data.columns.tolist())
        
        # Cek apakah ada minimal 5 baris dan kolom ke-2 ada
        if len(data) >= 5 and len(data.columns) > 1:
            st.write(f"Sample label powerset: {data.iloc[:5, 1].tolist()}")
        else:
            st.write(f"⚠️  Data terlalu sedikit atau kolom kurang. Rows: {len(data)}, Cols: {len(data.columns)}")
    else:
        st.error("DataFrame kosong. Pastikan data berhasil dimuat.")
        
except Exception as e:
    st.error(f"Error dalam original logic: {str(e)}")
