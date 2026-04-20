import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text(" ")


# INFO:
# real job descriptions from personal job tracker
df1 = pd.read_csv("./jobs.csv")

# INFO:
# from kaggle https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description
df2 = pd.read_csv("./job_title_des.csv")
df2 = df2["Job Description"].apply(clean_html)

df = df1["description"].apply(clean_html)
df = pd.concat([df, df2], ignore_index=True)

vectorizer = TfidfVectorizer(
    token_pattern=r"[a-z0-9+#]+", lowercase=True, stop_words="english", min_df=2
)
X = vectorizer.fit_transform(df)
terms = vectorizer.get_feature_names_out()
idf = vectorizer.idf_

im = {term: float(weight) for term, weight in zip(terms, idf)}
# im = [[k, v] for k, v in im.items()]
# im.sort(key=lambda i: i[1])
