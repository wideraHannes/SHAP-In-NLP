from pathlib import Path
from sklearn.utils import shuffle
import os
import urllib.request
import tarfile

from bs4 import BeautifulSoup
import contractions
import re
import string


def download_data():
    """
    The function checks if the IMDb dataset is already downloaded and extracted in the destination folder. 
    If not, it downloads the dataset, extracts it, and returns a message confirming the completion.
    """
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    destination_folder = os.path.join(os.path.dirname(os.getcwd()), "data")

    if os.path.exists(destination_folder):
        return "Already downloaded and extracted."
    else:
        os.makedirs(destination_folder)

    filename, _ = urllib.request.urlretrieve(url)

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(destination_folder)

    os.remove(filename)
    return "IMDb dataset downloaded and extracted to the '{}' folder.".format(destination_folder)


def read_imdb_split(split_dir):
    """the function reads the IMDb dataset from a specified split directory. It iterates through both "pos" and "neg" subdirectories, 
    reads the text files, and stores the texts in a list. Additionally, it assigns binary labels (0 for "neg" and 1 for "pos"). 
    After shuffling the data, 
    it applies a preprocessor function to each text and returns the preprocessed texts and their corresponding labels as lists.
    """
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir == "neg" else 1)
    texts, labels= shuffle(texts, labels, random_state=42)
    texts = list(map(_preprocessor,texts))
    return texts, labels


def _preprocessor(text):
  soup = BeautifulSoup(text, 'html.parser')
  text = soup.get_text()
  text = text.lower()
  text = contractions.fix(text)
  text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
  text = re.sub(' +', ' ', text)
  return text
