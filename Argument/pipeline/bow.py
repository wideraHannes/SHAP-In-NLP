from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

import string
def tokenize_sentence(s):
    """ build customer tokenizer by lower case, lemmatize and remove stopwords """
    wordnet_lemmatizer = WordNetLemmatizer()
    # lower case
    s = s.lower()
    # strip punctuation
    s = s.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    # split string into words (tokens)
    tokens = s.split()
    # remove short words, they're probably not useful
    tokens = [t for t in tokens if len(t) > 2]
    # put words into base form
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def build_bow_1():
  print('Create new')
  english_stopwords = stopwords.words('english')
  selected_stopwords = english_stopwords[:115] # after 115, the words indicate sentiment
  vectorizer = CountVectorizer(tokenizer=tokenize_sentence, stop_words=selected_stopwords, max_features=1)
  return vectorizer