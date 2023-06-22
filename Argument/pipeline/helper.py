import pandas as pd

def load_csv():
    # URL of the IMDb dataset tar.gz file
    data = pd.read_csv("../data/train.csv")
    print(data.head())


load_csv()