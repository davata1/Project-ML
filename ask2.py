import streamlit as st
import pandas as pd
import numpy as np
import string
from itertools import chain
from tqdm.auto import tqdm
import re, string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

st.title('Prediksi tweet covid 19')
text = st.text_input("Masukkan teks")
button=st.button('Hasil Prediksi')

