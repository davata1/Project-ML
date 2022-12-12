import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

st.title("PENAMBANGAN DATA")
st.write("##### Nama  :  ")
st.write("##### Nim   :  ")
st.write("##### Kelas :  ")

#Navbar
data_set_description, upload_data, preprocessing, modeling, implementation = st.tabs(["Data Set Description", "Upload Data", "Preprocessing", "Modeling", "Implementation"])

df = pd.read_csv('https://raw.githubusercontent.com/davata1/Datamining/main/final_test.csv')

#data_set_description
with data_set_description:
    st.write("###### Data Set Ini Adalah : Weather Prediction (Prediksi Cuaca) ")
    st.write("###### Sumber Data Set dari Kaggle : https://www.kaggle.com/datasets/ananthr1/weather-prediction")
    st.write("""1. preciptation (curah hujan) :
    Curah hujan : jumlah hujan yang turun pada suatu daerah dalam waktu tertentu. untuk menentukan besarnya curah hujan, membutuhkan suatu alat ukur. Alat pengukur curah hujan disebut dengan fluviograf dan satuan curah hujan yang biasanya digunakan adalah milimeter (mm).
    """)
    st.write("""2. tempmax (suhu maks) :
    Suhu Maksimum : Suhu yang terbaca dari termometer maksimum di ada di dataset
    """)
    st.write("""3. tempmin (suhu min) :
    Suhu Minimum : Suhu yang terbaca dari termometer minimum di ada di dataset
    """)
    st.write("""4. wind (angin) :
    Kecepatan angin disebabkan oleh pergerakan angin dari tekanan tinggi ke tekanan rendah, biasanya karena perubahan suhu
    """)
    st.write("""5. weather (cuaca) :
    Output (keluaran)
    """)
    st.write("""Menggunakan Kolom (input) :
    precipitation
    tempmax * tempmin
    wind
    """)
    st.write("""Memprediksi kondisi cuaca (output) :
    1. drizzle (gerimis)
    2. rain (hujan)
    3. sun (matahari)
    4. snow (salju)
    5. fog (kabut)
    """)
    st.write("###### Aplikasi ini untuk : Weather Prediction (Prediksi Cuaca) ")
    st.write("###### Source Code Aplikasi ada di Github anda bisa acces di link : ")

#Uploud data
with upload_data:
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        # view dataset asli
        st.header("Dataset")
        st.dataframe(df)

#Preprocessing
with preprocessing:
    st.subheader("""Normalisasi Data""")
    st.write("""Rumus Normalisasi Data :""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    #Mendefinisikan Varible X dan Y
    X = df.drop(columns=['size'])
    y = df['size'].values
    df_min = X.min()
    df_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.size).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        'S' : [dumies[0]],
        'M' : [dumies[1]],
        'L' : [dumies[2]],
        'XXS' : [dumies[3]],
        'Xl' : [dumies[4]],
        'XXXL' : [dumies[5]]
    })

    st.write(labels)

#Modelling
with modeling:
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        submitted = st.form_submit_button("Submit")

        # NB
        GaussianNB(priors=None)

        # Fitting Naive Bayes Classification to the Training set with linear kernel
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        y_pred = gaussian.predict(test)
    
        y_compare = np.vstack((test_label,y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))
        # akurasi = 10

        #KNN
        K=10
        knn=KNeighborsClassifier(n_neighbors=K)
        knn.fit(training,training_label)
        knn_predict=knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

        #Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        #Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        if submitted :
            if naive :
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if k_nn :
                st.write("Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree :
                st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
        
        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)

#Implementasi
with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        Precipitation = st.number_input('Masukkan preciptation (curah hujan) : ')
        Temp_Max = st.number_input('Masukkan tempmax (suhu maks) : ')
        Temp_Min = st.number_input('Masukkan tempmin (suhu min) : ')
        Wind = st.number_input('Masukkan wind (angin) : ')
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
                Precipitation,
                Temp_Max,
                Temp_Min,
                Wind
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if model == 'Gaussian Naive Bayes':
                mod = gaussian
            if model == 'K-NN':
                mod = knn 
            if model == 'Decision Tree':
                mod = dt

            input_pred = mod.predict(input_norm)

            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)
