from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def get_tfidf_vectorizer(train=None):
    tfidf = TfidfVectorizer(max_features=10000, strip_accents='unicode', analyzer='word')
    if train is not None:
        tfidf.fit(train)
    return tfidf


def normalize_sum_to_one(data):
    rowsums = data.sum(1).reshape(-1, 1)
    new_data = data / np.tile(rowsums, (1, data.shape[1]))
    return new_data
