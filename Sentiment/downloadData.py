import os
import urllib.request
import tarfile

# URL of the IMDb dataset tar.gz file
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# Destination folder for the extracted dataset
destination_folder =destination_folder = os.path.join("Sentiment", "data")

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Download the tar.gz file
filename, _ = urllib.request.urlretrieve(url)

# Extract the contents of the tar.gz file
with tarfile.open(filename, "r:gz") as tar:
    tar.extractall(destination_folder)

# Remove the downloaded tar.gz file
os.remove(filename)

print("IMDb dataset downloaded and extracted to the '{}' folder.".format(destination_folder))