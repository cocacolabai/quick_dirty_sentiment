import pandas as pd


# Read train.csv

train = pd.read_csv("data/train.csv")
corpus = []

for p in train.Phrase:
	corpus.append(p)

