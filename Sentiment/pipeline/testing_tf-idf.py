from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
import string
import re
import contractions

def preprocessor(text):
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = contractions.fix(text)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(' +', ' ', text)
    return text

  


text_1 = "I. like. this...      movie."
text_2 = "I hate this movie !!!!!"
text_3 =  "I don't <p> know what </p> to say about this movie"
print(preprocessor(text_1))
print(preprocessor(text_2))
print(preprocessor(text_3))