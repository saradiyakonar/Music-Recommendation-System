# preprocess.py
import pandas as pd
import numpy as np
import spacy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("songrec/songdata.csv")
df = df.sample(n=5000).drop('link',axis=1).reset_index(drop=True)

nlp = spacy.load("en_core_web_sm")


def tokenisation(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])


df["text"] = df["artist"] + " " + df["song"] 
df["text"] = df["text"].astype(str).apply(tokenisation)


vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(df["text"])


similarity = cosine_similarity(matrix)


with open("df.pkl", "wb") as f:
    pickle.dump(df, f)

with open("similarity.pkl", "wb") as f:
    pickle.dump(similarity, f)

print("Preprocessing complete. Pickles saved.")
