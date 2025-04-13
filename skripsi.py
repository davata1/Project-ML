import pandas as pd
import numpy as np
import re
import string
import nltk
import streamlit as st
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# Custom simple tokenizer function (not relying on NLTK punkt)
def simple_tokenize(text):
    if not isinstance(text, str):
        return []
    # Remove extra spaces and split by spaces
    return [word for word in re.sub(r'\s+', ' ', text).strip().split(' ') if word]

# Try to load stopwords, but provide a fallback if unavailable
try:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('indonesian'))
except:
    # Fallback Indonesian stopwords if NLTK download fails
    STOPWORDS = set([
        'ada', 'adalah', 'adanya', 'adapun', 'agak', 'agaknya', 'agar', 'akan', 'akankah', 'akhir',
        'akhiri', 'akhirnya', 'aku', 'akulah', 'amat', 'amatlah', 'anda', 'andalah', 'antar', 'antara',
        'antaranya', 'apa', 'apaan', 'apabila', 'apakah', 'apalagi', 'apatah', 'artinya', 'asal',
        'asalkan', 'atas', 'atau', 'ataukah', 'ataupun', 'awal', 'awalnya', 'bagai', 'bagaikan',
        'bagaimana', 'bagaimanakah', 'bagaimanapun', 'bagi', 'bagian', 'bahkan', 'bahwa', 'bahwasanya',
        'baik', 'bakal', 'bakalan', 'balik', 'banyak', 'bapak', 'baru', 'bawah', 'beberapa', 'begini',
        'beginian', 'beginikah', 'beginilah', 'begitu', 'begitukah', 'begitulah', 'begitupun', 'bekerja',
        'belakang', 'belakangan', 'belum', 'belumlah', 'benar', 'benarkah', 'benarlah', 'berada',
        'berakhir', 'berakhirlah', 'berakhirnya', 'berapa', 'berapakah', 'berapalah', 'berapapun',
        'berarti', 'berawal', 'berbagai', 'berdatangan', 'beri', 'berikan', 'berikut', 'berikutnya',
        'berjumlah', 'berkali-kali', 'berkata', 'berkehendak', 'berkeinginan', 'berkenaan', 'berlainan',
        'berlalu', 'berlangsung', 'berlebihan', 'bermacam', 'bermacam-macam', 'bermaksud', 'bermula',
        'bersama', 'bersama-sama', 'bersiap', 'bersiap-siap', 'bertanya', 'bertanya-tanya', 'berturut',
        'berturut-turut', 'bertutur', 'berujar', 'berupa', 'besar', 'betul', 'betulkah', 'biasa',
        'biasanya', 'bila', 'bilakah', 'bisa', 'bisakah', 'boleh', 'bolehkah', 'bolehlah', 'buat',
        'bukan', 'bukankah', 'bukanlah', 'bukannya', 'bulan', 'bung', 'cara', 'caranya', 'cukup',
        'cukupkah', 'cukuplah', 'cuma', 'dahulu', 'dalam', 'dan', 'dapat', 'dari', 'daripada', 'datang',
        'dekat', 'demi', 'demikian', 'demikianlah', 'dengan', 'depan', 'di', 'dia', 'diakhiri', 'diakhirinya',
        'dialah', 'diantara', 'diantaranya', 'diberi', 'diberikan', 'diberikannya', 'dibuat', 'dibuatnya',
        'didapat', 'didatangkan', 'digunakan', 'diibaratkan', 'diibaratkannya', 'diingat', 'diingatkan',
        'diinginkan', 'dijawab', 'dijelaskan', 'dijelaskannya', 'dikarenakan', 'dikatakan', 'dikatakannya',
        'dikerjakan', 'diketahui', 'diketahuinya', 'dikira', 'dilakukan', 'dilalui', 'dilihat', 'dimaksud',
        'dimaksudkan', 'dimaksudkannya', 'dimaksudnya', 'diminta', 'dimintai', 'dimisalkan', 'dimulai',
        'dimulailah', 'dimulainya', 'dimungkinkan', 'dini', 'dipastikan', 'diperbuat', 'diperbuatnya',
        'dipergunakan', 'diperkirakan', 'diperlihatkan', 'diperlukan', 'diperlukannya', 'dipersoalkan',
        'dipertanyakan', 'dipunyai', 'diri', 'dirinya', 'disampaikan', 'disebut', 'disebutkan',
        'disebutkannya', 'disini', 'disinilah', 'ditambahkan', 'ditandaskan', 'ditanya', 'ditanyai',
        'ditanyakan', 'ditegaskan', 'ditujukan', 'ditunjuk', 'ditunjuki', 'ditunjukkan', 'ditunjukkannya',
        'ditunjuknya', 'dituturkan', 'dituturkannya', 'diucapkan', 'diucapkannya', 'diungkapkan',
        'dong', 'dua', 'dulu', 'empat', 'enak', 'enggak', 'enggaknya', 'entah', 'entahlah', 'guna',
        'gunakan', 'hal', 'hampir', 'hanya', 'hanyalah', 'hari', 'harus', 'haruslah', 'harusnya',
        'hendak', 'hendaklah', 'hendaknya', 'hingga', 'ia', 'ialah', 'ibarat', 'ibaratkan', 'ibaratnya',
        'ibu', 'ikut', 'ingat', 'ingat-ingat', 'ingin', 'inginkah', 'inginkan', 'ini', 'inikah',
        'inilah', 'itu', 'itukah', 'itulah', 'jadi', 'jadilah', 'jadinya', 'jangan', 'jangankan',
        'janganlah', 'jauh', 'jawab', 'jawaban', 'jawabnya', 'jelas', 'jelaskan', 'jelaslah', 'jelasnya',
        'jika', 'jikalau', 'juga', 'jumlah', 'jumlahnya', 'justru', 'kala', 'kalau', 'kalaulah',
        'kalaupun', 'kalian', 'kami', 'kamilah', 'kamu', 'kamulah', 'kan', 'kapan', 'kapankah',
        'kapanpun', 'karena', 'karenanya', 'kasus', 'kata', 'katakan', 'katakanlah', 'katanya', 'ke',
        'keadaan', 'kebetulan', 'kecil', 'kedua', 'keduanya', 'keinginan', 'kelamaan', 'kelihatan',
        'kelihatannya', 'kelima', 'keluar', 'kembali', 'kemudian', 'kemungkinan', 'kemungkinannya',
        'kenapa', 'kepada', 'kepadanya', 'kesampaian', 'keseluruhan', 'keseluruhannya', 'keterlaluan',
        'ketika', 'khususnya', 'kini', 'kinilah', 'kira', 'kira-kira', 'kiranya', 'kita', 'kitalah',
        'kok', 'kurang', 'lagi', 'lagian', 'lah', 'lain', 'lainnya', 'lalu', 'lama', 'lamanya', 'lanjut',
        'lanjutnya', 'lebih', 'lewat', 'lima', 'luar', 'macam', 'maka', 'makanya', 'makin', 'malah',
        'malahan', 'mampu', 'mampukah', 'mana', 'manakala', 'manalagi', 'masa', 'masalah', 'masalahnya',
        'masih', 'masihkah', 'masing', 'masing-masing', 'mau', 'maupun', 'melainkan', 'melakukan',
        'melalui', 'melihat', 'melihatnya', 'memang', 'memastikan', 'memberi', 'memberikan', 'membuat',
        'memerlukan', 'memihak', 'meminta', 'memintakan', 'memisalkan', 'memperbuat', 'mempergunakan',
        'memperkirakan', 'memperlihatkan', 'mempersiapkan', 'mempersoalkan', 'mempertanyakan',
        'mempunyai', 'memulai', 'memungkinkan', 'menaiki', 'menambahkan', 'menandaskan', 'menanti',
        'menanti-nanti', 'menantikan', 'menanya', 'menanyai', 'menanyakan', 'mendapat', 'mendapatkan',
        'mendatang', 'mendatangi', 'mendatangkan', 'menegaskan', 'mengakhiri', 'mengapa', 'mengatakan',
        'mengatakannya', 'mengenai', 'mengerjakan', 'mengetahui', 'menggunakan', 'menghendaki',
        'mengibaratkan', 'mengibaratkannya', 'mengingat', 'mengingatkan', 'menginginkan', 'mengira',
        'mengucapkan', 'mengucapkannya', 'mengungkapkan', 'menjadi', 'menjawab', 'menjelaskan',
        'menuju', 'menunjuk', 'menunjuki', 'menunjukkan', 'menunjuknya', 'menurut', 'menuturkan',
        'menyampaikan', 'menyangkut', 'menyatakan', 'menyebutkan', 'menyeluruh', 'menyiapkan',
        'merasa', 'mereka', 'merekalah', 'merupakan', 'meski', 'meskipun', 'meyakini', 'meyakinkan',
        'minta', 'mirip', 'misal', 'misalkan', 'misalnya', 'mula', 'mulai', 'mulailah', 'mulanya',
        'mungkin', 'mungkinkah', 'nah', 'naik', 'namun', 'nanti', 'nantinya', 'nyaris', 'nyatanya',
        'oleh', 'olehnya', 'pada', 'padahal', 'padanya', 'pak', 'paling', 'panjang', 'pantas',
        'para', 'pasti', 'pastilah', 'penting', 'pentingnya', 'per', 'percuma', 'perlu', 'perlukah',
        'perlunya', 'pernah', 'persoalan', 'pertama', 'pertama-tama', 'pertanyaan', 'pertanyakan',
        'pihak', 'pihaknya', 'pukul', 'pula', 'pun', 'punya', 'rasa', 'rasanya', 'rata', 'rupanya',
        'saat', 'saatnya', 'saja', 'sajalah', 'saling', 'sama', 'sama-sama', 'sambil', 'sampai',
        'sampai-sampai', 'sampaikan', 'sana', 'sangat', 'sangatlah', 'satu', 'saya', 'sayalah',
        'se', 'sebab', 'sebabnya', 'sebagai', 'sebagaimana', 'sebagainya', 'sebagian', 'sebaik',
        'sebaik-baiknya', 'sebaiknya', 'sebaliknya', 'sebanyak', 'sebegini', 'sebegitu', 'sebelum',
        'sebelumnya', 'sebenarnya', 'seberapa', 'sebesar', 'sebetulnya', 'sebisanya', 'sebuah',
        'sebut', 'sebutlah', 'sebutnya', 'secara', 'secukupnya', 'sedang', 'sedangkan', 'sedemikian',
        'sedikit', 'sedikitnya', 'seenaknya', 'segala', 'segalanya', 'segera', 'seharusnya',
        'sehingga', 'seingat', 'sejak', 'sejauh', 'sejenak', 'sejumlah', 'sekadar', 'sekadarnya',
        'sekali', 'sekali-kali', 'sekalian', 'sekaligus', 'sekalipun', 'sekarang', 'sekarang',
        'sekecil', 'seketika', 'sekiranya', 'sekitar', 'sekitarnya', 'sekurang-kurangnya',
        'sekurangnya', 'sela', 'selain', 'selaku', 'selalu', 'selama', 'selama-lamanya',
        'selamanya', 'selanjutnya', 'seluruh', 'seluruhnya', 'semacam', 'semakin', 'semampu',
        'semampunya', 'semasa', 'semasih', 'semata', 'semata-mata', 'semaunya', 'sementara',
        'semisal', 'semisalnya', 'sempat', 'semua', 'semuanya', 'semula', 'sendiri', 'sendirian',
        'sendirinya', 'seolah', 'seolah-olah', 'seorang', 'sepanjang', 'sepantasnya', 'sepantasnyalah',
        'seperlunya', 'seperti', 'sepertinya', 'sepihak', 'sering', 'seringnya', 'serta', 'serupa',
        'sesaat', 'sesama', 'sesampai', 'sesegera', 'sesekali', 'seseorang', 'sesuatu', 'sesuatunya',
        'sesudah', 'sesudahnya', 'setelah', 'setempat', 'setengah', 'seterusnya', 'setiap', 'setiba',
        'setibanya', 'setidak-tidaknya', 'setidaknya', 'setinggi', 'seusai', 'sewaktu', 'siap',
        'siapa', 'siapakah', 'siapapun', 'sini', 'sinilah', 'soal', 'soalnya', 'suatu', 'sudah',
        'sudahkah', 'sudahlah', 'supaya', 'tadi', 'tadinya', 'tahu', 'tahun', 'tak', 'tambah',
        'tambahnya', 'tampak', 'tampaknya', 'tandas', 'tandasnya', 'tanpa', 'tanya', 'tanyakan',
        'tanyanya', 'tapi', 'tegas', 'tegasnya', 'telah', 'tempat', 'tengah', 'tentang', 'tentu',
        'tentulah', 'tentunya', 'tepat', 'terakhir', 'terasa', 'terbanyak', 'terdahulu', 'terdapat',
        'terdiri', 'terhadap', 'terhadapnya', 'teringat', 'teringat-ingat', 'terjadi', 'terjadilah',
        'terjadinya', 'terkira', 'terlalu', 'terlebih', 'terlihat', 'termasuk', 'ternyata', 'tersampaikan',
        'tersebut', 'tersebutlah', 'tertentu', 'tertuju', 'terus', 'terutama', 'tetap', 'tetapi',
        'tiap', 'tiba', 'tiba-tiba', 'tidak', 'tidakkah', 'tidaklah', 'tiga', 'tinggi', 'toh',
        'tunjuk', 'turut', 'tutur', 'tuturnya', 'ucap', 'ucapnya', 'ujar', 'ujarnya', 'umum',
        'umumnya', 'ungkap', 'ungkapnya', 'untuk', 'usah', 'usai', 'waduh', 'wah', 'wahai',
        'waktu', 'waktunya', 'walau', 'walaupun', 'wong', 'yaitu', 'yakin', 'yakni', 'yang', 'yg'
    ])

# Load data function with error handling
@st.cache_data
def load_data():
    try:
        return pd.read_excel('https://github.com/davata1/Project-ML/raw/refs/heads/main/data%20fix.xlsx')
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load normalization dictionary
@st.cache_data
def load_normalization_dict():
    try:
        kamus = pd.read_csv('https://github.com/davata1/Project-ML/raw/refs/heads/main/colloquial-indonesian-lexicon.csv')
        return kamus.drop(columns=['In-dictionary','context','category1','category2', 'category3'])
    except Exception as e:
        st.error(f"Error loading normalization dictionary: {str(e)}")
        return None

# Text preprocessing functions
def case_fold(text):
    return text.lower() if isinstance(text, str) else ""

def remove_punctuation(text):
    if not isinstance(text, str):
        return ""
    data = re.sub('@[^\s]+', ' ', text)
    data = re.sub(r'http\S*', ' ', data)
    data = data.translate(str.maketrans(' ', ' ', string.punctuation))
    data = re.sub('[^a-zA-Z]', ' ', data)
    data = re.sub("\n", " ", data)
    data = re.sub(r"\b[a-zA-z]\b", " ", data)
    return data

def normalization(token, kamus_normalisasi):
    ulasan_normalisasi = []
    for kata in token:
        if kata in kamus_normalisasi['slang'].values:
            formal = kamus_normalisasi.loc[kamus_normalisasi['slang'] == kata, 'formal'].values[0]
            ulasan_normalisasi.append(formal)
        else:
            ulasan_normalisasi.append(kata)
    return ulasan_normalisasi

# Create stemmer
try:
    Fact = StemmerFactory()
    Stemmer = Fact.create_stemmer()

    def stemming(ulasan):
        result = []
        for word in ulasan:
            result.append(Stemmer.stem(word))
        return result
except:
    # Basic fallback stemmer that just removes common Indonesian suffixes
    def stemming(ulasan):
        result = []
        for word in ulasan:
            # Simple suffix removal (not as good as Sastrawi but better than nothing)
            for suffix in ['kan', 'an', 'i', 'lah', 'kah', 'nya', 'pun']:
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    word = word[:-len(suffix)]
                    break
            result.append(word)
        return result

def remove_stopword(ulasan):
    result = []
    for word in ulasan:
        if word not in STOPWORDS:
            result.append(word)
    return result

# Train model function
@st.cache_resource
def train_model(X_train, y_train):
    try:
        with st.spinner('Training model... Please wait...'):
            # Create and train the model
            svm = SVC(kernel='linear', probability=True)
            model = MultiOutputClassifier(svm)
            model.fit(X_train, y_train)
            return model
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

# Main application
def main():
    with st.container():
        st.title('Aplikasi Analisis Sentimen Berbasis Aspek :red[Ulasan Rumah Makan Bebek Sinjay]')
        
        # Show data loading progress
        with st.spinner('Loading data and preparing model...'):
            # Load data and resources
            data = load_data()
            kamus_normalisasi = load_normalization_dict()
            
            if data is None or kamus_normalisasi is None:
                st.error("Failed to load necessary resources. Please check your internet connection.")
                return
            
            # Process training data
            st.session_state.loading_status = st.empty()
            
            if 'data_processed' not in st.session_state:
                st.session_state.loading_status.info("Processing training data...")
                
                try:
                    data['case_folding'] = data['ulasan'].apply(case_fold)
                    data['clean'] = data['case_folding'].apply(remove_punctuation)
                    data['tokenisasi'] = data['clean'].apply(simple_tokenize)  # Use our simple tokenizer
                    data['normal'] = data['tokenisasi'].apply(lambda x: normalization(x, kamus_normalisasi))
                    data['stemming'] = data['normal'].apply(stemming)
                    data['stopword'] = data['stemming'].apply(remove_stopword)
                    data['final'] = data['stopword'].apply(lambda tokens: ' '.join(tokens))
                    
                    X = data['final'].values.tolist()
                    y = np.asarray(data[data.columns[1:11]])
                    
                    # Vectorize training data
                    st.session_state.loading_status.info("Vectorizing data...")
                    vectorizer = TfidfVectorizer(max_features=2500, max_df=0.9)
                    tfidf_vectors = vectorizer.fit_transform(X)
                    X_train = tfidf_vectors.toarray()
                    
                    # Store in session state
                    st.session_state.vectorizer = vectorizer
                    st.session_state.X_train = X_train
                    st.session_state.y_train = y
                    st.session_state.data_processed = True
                    
                    # Train model
                    st.session_state.loading_status.info("Training model... This might take a few minutes...")
                    model = train_model(X_train, y)
                    st.session_state.model = model
                    
                    if model is not None:
                        st.session_state.loading_status.success("Model ready!")
                    else:
                        st.session_state.loading_status.error("Failed to train model.")
                        return
                except Exception as e:
                    st.error(f"Error during data processing: {str(e)}")
                    return
            else:
                # Use stored data and model
                st.session_state.loading_status.success("Model ready!")
        
        input_text = st.text_area("**Masukkan Ulasan**")
        submit = st.button("Proses", type="primary")

        if submit:
            if not input_text.strip():
                st.warning("Mohon masukkan ulasan terlebih dahulu.")
                return
                
            try:
                # Create dataframe from input
                df_mentah = pd.DataFrame({'Ulasan': [input_text]})
                
                # Process input data
                df_mentah['case_folding'] = df_mentah['Ulasan'].apply(case_fold)
                df_mentah['clean'] = df_mentah['case_folding'].apply(remove_punctuation)
                df_mentah['tokenisasi'] = df_mentah['clean'].apply(simple_tokenize)  # Use our simple tokenizer
                df_mentah['normal'] = df_mentah['tokenisasi'].apply(lambda x: normalization(x, kamus_normalisasi))
                df_mentah['stemming'] = df_mentah['normal'].apply(stemming)
                df_mentah['stopword'] = df_mentah['stemming'].apply(remove_stopword)
                df_mentah['final'] = df_mentah['stopword'].apply(lambda tokens: ' '.join(tokens))
                
                # Transform input
                vectorizer = st.session_state.vectorizer
                transform = vectorizer.transform(df_mentah['final'])
                transform_dense = np.asarray(transform.todense())
                
                # Make prediction
                model = st.session_state.model
                pred = model.predict(transform_dense)
                
                # Display results
                st.write("**Aspek dan sentimen yang terkandung dalam ulasan**")
                
                nama_fitur = [
                    ['Makanan','Positif'],['Makanan','Negatif'],
                    ['Layanan','Positif'],['Layanan','Negatif'],
                    ['Tempat','Positif'],['Tempat','Negatif'],
                    ['Harga','Positif'],['Harga','Negatif'],
                    ['Lainnya','Positif'],['Lainnya','Negatif']
                ]
                
                # Dictionary untuk warna background berdasarkan aspek
                aspek_warna = {
                    'Makanan': '#fecdd3',  # pink
                    'Layanan': '#fde68a',  # kuning
                    'Tempat': '#a7f3d0',   # hijau
