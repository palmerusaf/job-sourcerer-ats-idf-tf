import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("./jobs.csv")
vectorizer = TfidfVectorizer(
    lowercase=True, stop_words="english", ngram_range=(1, 2), max_df=0.85, min_df=2
)


def clean_html(text):
    if not isinstance(text, str):
        return ""
    return BeautifulSoup(text, "html.parser").get_text(" ")


ds = df["description"].apply(clean_html)
print(ds.head())
X = vectorizer.fit_transform(ds)
# TODO: transfrom to json <string,num>
