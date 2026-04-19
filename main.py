import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("./jobs.csv")
vectorizer = TfidfVectorizer(
    lowercase=True, stop_words="english", max_df=0.85, min_df=2
)


def clean_html(text):
    if not isinstance(text, str):
        return ""
    return BeautifulSoup(text, "html.parser").get_text(" ")


ds = df["description"].apply(clean_html)
X = vectorizer.fit_transform(ds)
terms = vectorizer.get_feature_names_out()
idf = vectorizer.idf_

idf_map = {term: float(weight) for term, weight in zip(terms, idf)}
