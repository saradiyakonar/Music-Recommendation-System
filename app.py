# app.py
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)


df = pickle.load(open("df.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))

def recommend(song_name):
    if song_name not in df['song'].values:
        return ["Song not found in dataset."]
    idx = df[df['song'] == song_name].index[0]
    distances = list(enumerate(similarity[idx]))
    distances = sorted(distances, reverse=True, key=lambda x: x[1])
    recommended = [df.iloc[i[0]]['song'] + " by " + df.iloc[i[0]]['artist'] for i in distances[1:6]]
    return recommended

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    query = ""
    if request.method == "POST":
        query = request.form["song"]
        recommendations = recommend(query)
    return render_template("index.html", recommendations=recommendations, query=query)

if __name__ == "__main__":
    app.run(debug=True)
