import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, strip_accents_unicode
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
        'min_df': 5,
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


def clean_string(string):
    new_string = string.lower()  # Also done by the vectorizer, but do it here anyway
    # remove RT
    new_string = re.sub(r'\brt\b', '', new_string)
    # remove @mention
    new_string = re.sub(r'@mention', '', new_string)
    # remove links, http and {link}
    new_string = re.sub(r'\bhttp\S*\b', '', new_string)
    new_string = re.sub(r'\{link\}', '', new_string)
    # Need to convert to list of words for the stemmer
    token_pattern = r'\b[a-zA-Z]{2,}\b'
    vectorizer = re.compile(token_pattern)
    words = vectorizer.findall(new_string)
    stemmer = nltk.PorterStemmer()
    words = [stemmer.stem(x) for x in words]
    new_string = ' '.join(words)
    return new_string
