import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from skmultilearn.problem_transform import LabelPowerset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss, f1_score
import streamlit as st
# Load dataset
df = pd.read_excel("/content/drive/MyDrive/SKRIPSI/dataset/dataset skripsi/sinjaymadura.xlsx")

# Split label jadi list
df['label'] = df['label'].apply(lambda x: x.split(','))

# MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['label'])

# TF-IDF fitur dari teks
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Pre Processing'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Total data
total_data = X.shape[0]

# Jumlah data train dan test
jumlah_train = X_train.shape[0]
jumlah_test = X_test.shape[0]

# Cetak bukti
print(f"Total data     : {total_data}")
print(f"Data train     : {jumlah_train} ({(jumlah_train/total_data)*100:.2f}%)")
print(f"Data test      : {jumlah_test} ({(jumlah_test/total_data)*100:.2f}%)")

# Skenario 1
skenario_1 = {"learning_rate": 0.1, "reg_lambda": 1, "gamma": 0}

# Model
model = LabelPowerset(
    XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=5,
        learning_rate=skenario_1['learning_rate'],
        reg_lambda=skenario_1['reg_lambda'],
        gamma=skenario_1['gamma']
    )
)

# Training
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
h_loss = hamming_loss(y_test, y_pred)
micro_f1 = f1_score(y_test, y_pred, average='micro')

print("Hamming Loss:", h_loss)
print("Micro F1 Score:", micro_f1)

# Tampilkan classification report
print(classification_report(y_test, y_pred, target_names=mlb.classes_))
