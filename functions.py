# -------------------------------------------------------
# LIBRARY
# -------------------------------------------------------

# Framework python untuk tampilan web
import streamlit as st
# Library untuk membuat plot grafik
import matplotlib.pyplot as plt
# Untuk olah data pada DataFrame
import pandas as pd
# Untuk berinteraksi dengan sistem operasi
import os
# Untuk pengolahan citra dan computer vision
import cv2
# Untuk metode k-Fold Cross Validation
from sklearn.model_selection import KFold
# Untuk membuat salinan file ke folder tujuan
import shutil
# Untuk menghasilkan nilai acak
import random
# Untuk komputasi matematika
import numpy as np
# Untuk labeling ke bentuk numerik pada kolom label/target
from sklearn.preprocessing import LabelEncoder
# Untuk mengelola iterasi dengan membuat iterator khusus
import itertools
# Untuk augmentasi data gambar
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Untuk menyimpan dan memuat data sebagai objek biner
import pickle
# Untuk evaluasi kinerja klasifikasi
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

# Disable warning
st.set_option("deprecation.showPyplotGlobalUse", False)

# -------------------------------------------------------
# FUNCTION
# -------------------------------------------------------

# Beberapa fungsi untuk menambahkan margin pada tampilan web
# Make Space
def ms_20():
    st.markdown(
        '''
        <div class="ms-20"></div>
        ''',
        unsafe_allow_html= True,
    )

def ms_40():
    st.markdown(
        '''
        <div class="ms-40"></div>
        ''',
        unsafe_allow_html= True,
    )

def ms_60():
    st.markdown(
        '''
        <div class="ms-60"></div>
        ''',
        unsafe_allow_html= True,
    )

def ms_80():
    st.markdown(
        '''
        <div class="ms-80"></div>
        ''',
        unsafe_allow_html= True,
    )

# Fungsi untuk layouting kolom
# Make Layout
def ml_main():
    left, center, right = st.columns([.3, 2.5, .3])
    return center

def ml_double():
    left, center, right = st.columns([1, .05, 1])
    return left, right

def ml_right():
    left, center, right = st.columns([1, .1, 1.5])
    return left, right

def ml_left():
    left, center, right = st.columns([1.5, .1, 1])
    return left, right

# Fungsi untuk menampilkan judul section
def prn_judul(judul, size= 3, line= False):
    h = "#" if size == 1 else (
            "##" if size == 2 else (
                "###" if size == 3 else (
                    "####" if size == 4 else "#####"
                )
            )
        )
    # Tampilkan judul
    st.write(f"{h} **{judul}**")
    if line:
        # Buat garis
        st.markdown("---")

# Fungsi untuk menampilkan caption section
def prn_caption(text, size= 3, line= False):
    h = "#" if size == 1 else (
            "##" if size == 2 else (
                "###" if size == 3 else (
                    "####" if size == 4 else "#####"
                )
            )
        )
    # Tampilkan judul
    st.caption(f"{h} **{text}**")
    if line:
        # Buat garis
        st.markdown("---")

# Fungsi untuk membaca dataset (etc: xlsx, csv)
def get_xlsx(filepath):
    return pd.read_excel(filepath)

def get_csv(filepath):
    return pd.read_csv(filepath)

# Fungsi untuk mendapatkan path image
def get_filepath(PATH, name_df):
    # Buat list untuk menyimpan informasi data gambar
    lst_path, lst_name, lst_label = list(), list(), list()
    # Loop untuk setiap folder gambar
    for folder in os.listdir(PATH):
        # Gabungkan PATH utama dengan PATH folder
        folderpath = os.path.join(PATH, folder)
        if os.path.isdir(folderpath):
            # Loop untuk setiap file gambar
            for file in os.listdir(folderpath):
                # Dapatkan PATH setiap file yg ada dalam folder
                filepath = os.path.join(folderpath, file)

                # Simpan informasi file yg didapatkan
                lst_path.append(filepath)
                lst_name.append(file)
                lst_label.append(folder)
    # Buat DataFrame dari informasi yg diambil
    data = pd.DataFrame(
        {
            "filepath": lst_path,
            "filename": lst_name,
            "label": lst_label,
        }
    )
    # Buat folder jika belum ada
    mk_dir("processed/dataframe")
    # Simpan DataFrame ke dalam lokal file
    save_df(data, f"processed/dataframe/{name_df}")
    return data

# Fungsi untuk membuat folder (direktori)
def mk_dir(dirname):
    # Cek apakah direktori tersedia
    if not os.path.exists(dirname):
        os.makedirs(dirname) # Buat direktorinya

# Fungsi untuk menyimpan dataframe
def save_df(df, filepath):
    df.to_csv(f"{filepath}.csv", index= False)

# Fungsi untuk menampilkan data gambar
@st.cache_data(ttl=3600, show_spinner="Show images...")  # 👈 Cache data
def show_images(df):
    # Ambil beberapa path gambar
    img_path = df["filepath"].head(12)

    # Inisialisasi figure plot
    plt.figure(figsize= (10, 10))
    # Perulangan untuk menampilkan setiap citra
    for idx, filepath in enumerate(img_path):
        # Baca data gambar
        img = cv2.imread(filepath)

        # Inisialisasi subplot
        plt.subplot(4, 4, idx + 1)
        # Tampilkan citra dalam plot figure
        plt.imshow(img)
        # Judul dan nama file
        plt.title(df.iloc[:, 1][idx], color="b", fontsize= 15)
        # Hapus garis figure
        plt.axis("off")
    # Tampilkan figure yg telah di isi citra
    st.pyplot()

# Function untuk menampilkan skor akurasi
def show_acc(test, pred):
    acc = accuracy_score(test, pred) * 100
    st.info(
        f"Skor akurasi: {acc:.2f}%"
    )
    return acc

# Fungsi untuk membuat confusion matrix
def create_confusion_matrix(test, pred):
    # Buat objek confusion matrix
    cm = confusion_matrix(test, pred)
    # Ambil unique label dari data test
    classes = np.unique(test)
    # Tampilkan confusion matrix-nya
    plot_confusion_matrix(
        cm= cm, 
        classes= classes, 
        title= "Confusion Matrix"
    )

# Fungsi untuk membuat classification report
def create_classification_report(test, pred):
    # Buat objek classification report
    st.dataframe(
        pd.DataFrame(
            classification_report(
                test,
                pred,
                target_names= np.unique(test),
                output_dict= True,
            )
        ).transpose(),
        use_container_width= True,
    )

# Function untuk plotting confusion matrix
def plot_confusion_matrix(
        cm, 
        classes, 
        normalize= False, 
        title= "Confusion Matrix", 
        cmap= plt.cm.Blues
):
    # Buat plot figure-nya
    plt.figure(figsize= (10, 10))
    plt.imshow(cm, interpolation= 'nearest', cmap= cmap)
    plt.title(title)
    plt.colorbar()

    # Beri label pada garis x dan y
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation= 45)
    plt.yticks(tick_marks, classes)

    # Cek kondisi normalisasi
    if normalize:
        cm = cm.astype("float") / cm.sum(axis= 1)[:, np.newaxis]
        st.write("Normalized Confusion Matrix")
    else:
        st.write("Confusion Matrix, without Normalization")

    # Threshold
    thresh = cm.max() / 2.
    # Bentuk plot confusion matrix-nya
    for i, j in itertools.product(
        range(cm.shape[0]), 
        range(cm.shape[1])
    ):
        plt.text(
            j, 
            i, 
            cm[i, j], 
            fontsize= 16, 
            horizontalalignment= "center", 
            color= "white" if cm[i, j] > thresh else "black"
        )

    plt.tight_layout()
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    st.pyplot()

# -------------------------------------------------------
    
# Fungsi untuk melakukan resize pada gambar
@st.cache_data(ttl=3600, show_spinner="Resize images is running...")  # 👈 Cache data
def resize_image(df, width= 300, height= 300):
    # Buat folder untuk menyimpan gambar hasil resize
    for label in df.iloc[:, 2].unique():
        if not os.path.exists(f"processed/resized/{label}"):
            os.makedirs(f"processed/resized/{label}")

    for PATH, filename, label in zip(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]):
        # Load data menggunakan filepath
        image = cv2.imread(PATH)
        # Inisialisasi ukuran yg diinginkan
        size_image = (width, height)
        # Lakukan proses resize
        image_resize = cv2.resize(image, size_image, interpolation= cv2.INTER_AREA)
        # Simpan citra yg telah di resize ke dalam lokal folder
        result_path = os.path.join(f"processed/resized/{label}", filename)
        cv2.imwrite(result_path, image_resize)

# Fungsi untuk augmentasi gambar
@st.cache_data(ttl=3600, show_spinner="Augmentation images is running...")  # 👈 Cache data
def augment_image(PATH, count_):
    # Buat objek ImageDataGenerator
    datagen = ImageDataGenerator(zoom_range= [0.7, 1.2])
    # Dapatkan daftar citra dalam PATH
    lst_img = os.listdir(PATH)
    # Acak daftar, agar citra yg diaugmentasi bersifat acak
    random.shuffle(lst_img)
    # Nilai dari jumlah kebutuhan data yg perlu di sintesis
    n_augment = count_ - len(lst_img)
    # Variabel bantu untuk indexing data citra
    idx = 0
    # Loop untuk melakukan augmentasi pada sejumlah data citra
    for i in range(n_augment):
        # Cek nilai indeks, jika melebihi total gambar
        # Maka ubah nilai index jadi 0
        if idx >= len(lst_img):
            idx = 0
        # Ambil nama gambar
        filename = lst_img[idx]
        # Ambil PATH citra
        filepath = os.path.join(PATH, filename)
        # Baca data gambar
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Ubah dimensi citra jadi (1, H, W, channels) karena ImageDataGenerator butuh bentuk batch
        img_batch = img.reshape((1,) + img.shape)
        # Buat generator augmentasi
        augmented_generator = datagen.flow(img_batch, batch_size= 1)
        # Buat gambar augmentasinya sebanyak 1
        augmented_img = [next(augmented_generator)[0].astype(np.uint8) for _ in range(1)]

        # Simpan gambar yg telah di Zoom
        cv2.imwrite(
            os.path.join(PATH, f"augmented_{i}_{filename}"),
            np.array(augmented_img[0]),
        )

        # Increment
        idx += 1

# Function untuk ekstraksi fitur HSV dari gambar
@st.cache_data(ttl=3600, show_spinner="Extraction features is running...")  # 👈 Cache data
def extraction_HSV_features(PATH, label):
    # Dapatkan daftar citra dalam folder PATH
    lst_img = os.listdir(PATH)
    # Inisialisasi list untuk menyimpan nilai fitur dan filename
    hue_ftr, saturation_ftr, value_ftr = list(), list(), list()
    filenames = list()

    # Loop untuk setiap gambar dalam lst_img
    for filename in lst_img:
        # Ambil path gambar
        filepath = os.path.join(PATH, filename)

        # Baca data gambar
        img = cv2.imread(filepath)
        # Konversi citra ke ruang warna HSV
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Dapatkan nilai median untuk setiap channel warna
        med_H = np.median(img_HSV[:, :, 0])
        med_S = np.median(img_HSV[:, :, 1])
        med_V = np.median(img_HSV[:, :, 2])

        # Simpan nilai median masing-masing channel ke dalam list terkait
        hue_ftr.append(med_H)
        saturation_ftr.append(med_S)
        value_ftr.append(med_V)

        # simpan filename citra
        filenames.append(filename)
    # Buat dataframe dari list yg telah terisi
    df = pd.DataFrame(
        {
            "filename": filenames,
            "hue": hue_ftr,
            "saturation": saturation_ftr,
            "value": value_ftr,
            "label": [label] * len(filenames),
        }
    )
    return df

# Function untuk menghitung jarak Euclidean antara dua titik point
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Function untuk perhitungan metode KNN
@st.cache_data(ttl=3600, show_spinner="Calculating distance...")  # 👈 Cache data
def train_model_KNN(X_train, y_train, X_test, neighbor):
    # Pengkodean pada kolom target
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    
    # Ambil jumlah data train dan test
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # Buat dummy data untuk kolom target
    y_pred = np.zeros(n_test)

    # Loop untuk menghitung jarak setiap data test
    for i in range(n_test):
        # Buat dummy data untuk nilai distance
        distances = np.zeros(n_train)
        # Loop untuk semua data train
        for j in range(n_train):
            # Hitung distance
            distances[j] = euclidean_distance(X_test[i], X_train[j])
        # Dapatkan indeks k tetangga terdekat
        nearest_indices = np.argsort(distances)[:neighbor]
        # Ambil nilai label dari data train yg termasuk tetangga terdekat
        nearest_labels = y_train_encoded[nearest_indices]

        # Hitung jumlah label unik yg muncul
        unique_labels, counts = np.unique(nearest_labels, return_counts= True)

        # Hasil prediksi target untuk label yg paling banyak muncul (voting)
        y_pred[i] = unique_labels[np.argmax(counts)]
    
    # Decode labelnya
    y_pred = [int(x) for x in y_pred]
    y_pred = encoder.inverse_transform(y_pred)
    return y_pred

# Function untuk split k fold
def fold_split(features, labels, k= 5, neighbor= 5):
    # Panggil objek KFold
    kfold = KFold(n_splits= k, shuffle= True, random_state= 42)
    
    # List untuk menyimpan data klasifikasi
    all_y_test, all_y_pred = list(), list()
    all_score = list()
    #all_X_test = list()

    # Loop KFold
    for fold, (tr_idx, ts_idx) in enumerate(
        kfold.split(features)
    ):
        # Inisialisasi data train dan data test
        X_train, X_test = features[tr_idx], features[ts_idx]
        y_train, y_test = labels[tr_idx], labels[ts_idx]

        # Perhitungan KNN
        y_pred = train_model_KNN(X_train, y_train, X_test, neighbor)

        # Simpan hasil prediksi dan label aktual dari data
        all_y_test.append(y_test)
        all_y_pred.append(y_pred)
        #all_X_test.extend(X_test)

        knn = KNeighborsClassifier(n_neighbors= neighbor)
        knn.fit(X_train, y_train)
        y_score = knn.predict_proba(X_test)

        all_score.append(y_score)
    
    mk_dir("processed/picklefile")
    # Menyimpan list ke dalam file lokal menggunakan pickle
    with open('processed/picklefile/all_y_test.pkl', 'wb') as file:
        pickle.dump(all_y_test, file)
    with open('processed/picklefile/all_y_pred.pkl', 'wb') as file:
        pickle.dump(all_y_pred, file)
    with open('processed/picklefile/all_score.pkl', 'wb') as file:
        pickle.dump(all_score, file)

# Function untuk menampilkan hasil prediksi
def show_predict():
    # Memuat list dari file lokal
    with open('processed/picklefile/all_y_test.pkl', 'rb') as file:
        all_y_test = pickle.load(file)
    with open('processed/picklefile/all_y_pred.pkl', 'rb') as file:
        all_y_pred = pickle.load(file)

    for fold, (y_test, y_pred) in enumerate(zip(all_y_test, all_y_pred)):
        with st.expander(f"Fold ke-{fold + 1}"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "aktual": y_test,
                        "prediksi": y_pred,
                    },
                ),
                use_container_width= True,
                hide_index= True,
            )