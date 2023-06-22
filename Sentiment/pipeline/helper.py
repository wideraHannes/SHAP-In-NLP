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
    # URL of the IMDb dataset tar.gz file
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    # Destination folder for the extracted dataset
    destination_folder = os.path.join(os.path.dirname(os.getcwd()), "data")

    # Create the destination folder if it doesn't exist
    if os.path.exists(destination_folder):
        return "Already downloaded and extracted."
    else:
        os.makedirs(destination_folder)

    # Download the tar.gz file
    filename, _ = urllib.request.urlretrieve(url)

    # Extract the contents of the tar.gz file
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(destination_folder)

    # Remove the downloaded tar.gz file
    os.remove(filename)
    return "IMDb dataset downloaded and extracted to the '{}' folder.".format(destination_folder)

def preprocessor(text):
  soup = BeautifulSoup(text, 'html.parser')
  text = soup.get_text()
  text = text.lower()
  text = contractions.fix(text)
  text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
  text = re.sub(' +', ' ', text)
  return text 

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir == "neg" else 1)
    texts, labels= shuffle(texts, labels, random_state=42)
    texts = list(map(preprocessor,texts))
    return texts, labels
