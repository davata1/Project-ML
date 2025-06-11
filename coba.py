import pickle
import pandas as pd
import numpy as np
import re
import string
import nltk
import streamlit as st

# Download NLTK data seperti di Google Colab
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from skmultilearn.problem_transform import LabelPowerset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss, f1_score
import joblib

# Memuat data dari file Excel
data = pd.read_excel('https://github.com/davata1/Project-ML/raw/refs/heads/main/sinjaymadura.xlsx')
data
# Cek apakah DataFrame tidak kosong
if not data.empty:
    st.write("DataFrame shape:", data.shape)
    st.write("DataFrame columns:", data.columns.tolist())
    st.write(f"Sample label powerset: {data.iloc[:5, 1].tolist()}")
else:
    st.error("DataFrame kosong. Pastikan data berhasil dimuat.")
