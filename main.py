import pandas as pd
from gensim.models import Word2Vec 
import logging
from sklearn import svm

# Configure loggin for gensim model training
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def buildVectors(train):
	corpus = []

	# Create corpus
	for p in train.Phrase:
		corpus.append(p.split())

	# Build word vectors
	model = Word2Vec(corpus, min_count=1, size=25)

	# Save model for later use (hdd persistant)
	model.save("model/model")

	return model


# Read train.csv
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

model = buildVectors(train)

pvecs = dict()

for r in train.iterrows():
	sid = r[1]["SentenceId"]
	phvec = sum([model[x] for x in r[1]["Phrase"].split()])
	pvecs[sid] = phvec


# Convert the dictionary into a dataframe
pvdf = pd.DataFrame.from_dict(pvecs, orient='index')

# Rename columns
pvdf.columns = ["feat_"+str(x) for x in range(1,26)]

# Add sentiment lable to pvdf
pvdf["label"] = train.Sentiment

# Define and train a classifier
clf = svm.SVC()
clf.fit(pvdf[pvdf.columns[:25]], pvdf.label)

# test.Phrase[0]
# 'An intermittently pleasing but mostly routine effort .'

# Lets create a phrase vector 
y = pd.DataFrame(sum([model[x] for x in test.Phrase[0].split()])).transpose()

# Make the final prediction
clf.predict(y)

# outputs 2 => neutral
