import nltk
from nltk.corpus import stopwords
from itertools import chain

nltk.download('stopwords')

stop_words = set(chain(stopwords.words('indonesian')))

def remove_stopwords(tokens):
    filtered_tokens = [w for w in tokens if not w in stop_words]
    return filtered_tokens
