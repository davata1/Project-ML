import streamlit as st
import pandas as pd
import numpy as np

st.title('Prediksi tweet covid 19')
text = st.text_input("Masukkan teks")
button=st.button('Hasil Prediksi')

data