import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

# real job descriptions from personal job tracker
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

im = {term: float(weight) for term, weight in zip(terms, idf)}
im = [[k, v] for k, v in im.items()]
im.sort(key=lambda i: i[1])
im = pd.DataFrame(im).tail(40)
# __AUTO_GENERATED_PRINT_VAR_START__
print(f" im: {str(im)}")  # __AUTO_GENERATED_PRINT_VAR_END__
