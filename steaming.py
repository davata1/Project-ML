from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()
def stem_text(text):
    stemmed_text = stemmer.stem(text)
    return stemmed_text