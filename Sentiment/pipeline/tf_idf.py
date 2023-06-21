from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
import contractions
import re

import string

def preprocessor(text):
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = text.lower()
    text = contractions.fix(text)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(' +', ' ', text)
    return text 

def tokenizer(text):
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]




def build_tf_idf():
  # im explainer wird der vectorizer anscheind eh nicht verwendet
  #  vectorizer = TfidfVectorizer(preprocessor=preprocessor, tokenizer=tokenizer, min_df=0.001, max_df=0.99)
  vectorizer = TfidfVectorizer(min_df=0.001, max_df=0.99)
  return vectorizer
