import nltk
from nltk.tokenize import word_tokenize
import pandas as pd

nltk.download('popular')

def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens
