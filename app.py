import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Fungsi untuk membuat dataset dengan jendela waktu
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), :-1]
        X.append(a)
        Y.append(dataset[i + time_step, -1])
    return np.array(X), np.array(Y)

# Fungsi untuk melatih model
def train_model():
    # Load data
    df = pd.read_excel('https://github.com/davata1/Project-ML/blob/main/Data%20TB.xls')

    # Transformasi fitur
    df['Jenis Kelamin'] = df['Jenis Kelamin'].replace('L', 1).replace('P', 0)
    df['Batuk Darah'] = df['Batuk Darah'].replace('IYA', 1).replace('TIDAK', 0)
    df['Sesak Nafas'] = df['Sesak Nafas'].replace('IYA', 1).replace('TIDAK', 0)
    df['Nafsu Makan'] = df['Nafsu Makan'].replace('Turun', 1).replace('Baik', 0)
    df['Hasil Pemeriksaan TCM '] = df['Hasil Pemeriksaan TCM '].replace('Rif Sen', 1).replace('Negative', 0)
    df['Diagnosis'] = df['Diagnosis'].replace('TBC SO', 1).replace('Bukan TBC', 0)

    # Pisahkan fitur biner dan kontinu
    binary_features = df[['Jenis Kelamin', 'Batuk Darah', 'Nafsu Makan', 'Sesak Nafas', 'Hasil Pemeriksaan TCM ', 'Diagnosis']]
    continuous_features = df[['Umur (Tahun)', 'BB']]

    # Normalisasi fitur kontinu
    scaler = MinMaxScaler(feature_range=(0, 1))
    continuous_features_scaled = scaler.fit_transform(continuous_features)

    # Gabungkan kembali fitur biner dan kontinu yang telah dinormalisasi
    data_processed = np.concatenate((continuous_features_scaled, binary_features.values), axis=1)

    # Terapkan oversampling dengan SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, Y_resampled = smote.fit_resample(data_processed, binary_features['Diagnosis'])

    # Ubah data menjadi bentuk yang sesuai untuk LSTM
    time_step = 10  # misalkan menggunakan 10 langkah waktu
    X, Y = create_dataset(X_resampled, time_step)

    # Bentuk ulang input untuk LSTM [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

    # Bagi data menjadi set pelatihan dan pengujian
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Latih model
    model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test), verbose=1)

    return model, X_test, Y_test, scaler

# Fungsi untuk menampilkan metrik evaluasi
def evaluate_model(model, X_test, Y_test):
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)

    st.write(f'Accuracy: {accuracy}')
    st.write(f'Precision: {precision}')
    st.write(f'Recall: {recall}')
    st.write(f'F1 Score: {f1}')

    # Hitung matriks confusion
    cm = confusion_matrix(Y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    st.pyplot()

# Streamlit app
st.title('Tuberculosis Diagnosis Prediction')

# Load atau latih model
model, X_test, Y_test, scaler = train_model()

# Input form
with st.form(key='input_form'):
    umur = st.number_input('Umur (Tahun)', min_value=0, max_value=120, value=25)
    jenis_kelamin = st.selectbox('Jenis Kelamin', options=[('Laki-laki', 1), ('Perempuan', 0)])
    bb = st.number_input('Berat Badan (kg)', min_value=0.0, max_value=200.0, value=60.0)
    batuk_darah = st.selectbox('Batuk Darah', options=[('IYA', 1), ('TIDAK', 0)])
    nafsu_makan = st.selectbox('Nafsu Makan', options=[('Turun', 1), ('Baik', 0)])
    sesak_nafas = st.selectbox('Sesak Nafas', options=[('IYA', 1), ('TIDAK', 0)])
    hasil_tcm = st.selectbox('Hasil Pemeriksaan TCM', options=[('Rif Sen', 1), ('Negative', 0)])
    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    # Proses input user
    user_data = np.array([
        umur, bb, 
        jenis_kelamin[1], batuk_darah[1], 
        nafsu_makan[1], sesak_nafas[1], 
        hasil_tcm[1]
    ]).reshape(1, -1)
    user_data_scaled = scaler.transform(user_data[:, :2])
    user_data_processed = np.concatenate((user_data_scaled, user_data[:, 2:]), axis=1)

    # Bentuk ulang input untuk LSTM
    user_data_reshaped = user_data_processed.reshape(1, 10, user_data_processed.shape[1])  # Assuming 10 time steps

    # Prediksi menggunakan model
    prediction = model.predict(user_data_reshaped)
    prediction_label = (prediction > 0.5).astype("int32")

    if prediction_label == 1:
        st.write('Predicted Diagnosis: TBC SO')
    else:
        st.write('Predicted Diagnosis: Bukan TBC')

# Evaluasi model
st.subheader('Model Evaluation')
evaluate_model(model, X_test, Y_test)
