from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def get_tfidf_vectorizer(train=None):
    tfidf = TfidfVectorizer(max_features=None, strip_accents='unicode', analyzer='word', ngram_range=(1, 3), lowercase=True, stop_words='english', min_df=5, max_df=0.5)
    if train is not None:
        tfidf.fit(train)
    return tfidf


def normalize_sum_to_one(data):
    rowsums = data.sum(1).reshape(-1, 1)
    new_data = data / np.tile(rowsums, (1, data.shape[1]))
    return new_data
