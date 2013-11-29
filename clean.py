from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def get_tfidf_vectorizer(train=None, **kwargs):

    # Michael: not sure where you used these parameters, as they ended up being hard coded into the ridge_003
    #vectorizer_args = {
    #    'max_features': None,
    #    'strip_accents': 'unicode',
    #    'analyzer': 'word',
    #    'ngram_range': (1, 3),
    #    'lowercase': True,
    #    'stop_words': 'english',
    #    'min_df': 5,
    #    'max_df': 0.5,
    #}

    vectorizer_args = {
        'max_features': 10000,
        'strip_accents': 'unicode',
        'analyzer': 'word',
    }

    vectorizer_args.update(kwargs)

    tfidf = TfidfVectorizer(**vectorizer_args)
    if train is not None:
        tfidf.fit(train)
    return tfidf


def normalize_sum_to_one(data):
    rowsums = data.sum(1).reshape(-1, 1)
    new_data = data / np.tile(rowsums, (1, data.shape[1]))
    return new_data
