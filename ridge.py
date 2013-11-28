import pandas as p
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from IPython import embed
import os

paths = ['data/train.csv', 'data/test.csv']
train_set = p.read_csv(paths[0])
test_set = p.read_csv(paths[1])
# print t #display the data

tfidf = TfidfVectorizer(max_features=10000, strip_accents='unicode', analyzer='word')
tfidf.fit(train_set['tweet'])
X = tfidf.transform(train_set['tweet'])
Xp = tfidf.transform(test_set['tweet'])
y = np.array(train_set.ix[:,4:])

def output(filename, predictions):
    prediction = np.array(np.hstack([np.matrix(test_set['id']).T, predictions]))
    col = '%i,' + '%f,'*23 + '%f'
    header = ','.join(['id'] + list(train_set.columns[4:])) + '\n'

    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, 'w') as f:
        f.write(header)
        np.savetxt(f, prediction, col, delimiter=',')

# One shot ridge regression
clf = linear_model.RidgeCV(alphas = np.linspace(0, 5, 10), cv = 5, normalize = True)
# clf = linear_model.Ridge(alpha = 0.5)
clf.fit(X, y)
test_prediction = clf.predict(Xp)
print 'Best Alpha:', clf.alpha_
print 'Train error: {0}'.format(np.sqrt(np.sum(np.array(np.array(clf.predict(X))-y)**2)/ (X.shape[0]*24.0)))
output('results/ridge.csv', test_prediction)

# Split the regression so that we regress for each set of hashtags
sentiment_cols = ['s1', 's2', 's3', 's4', 's5']
time_cols = ['w1', 'w2', 'w3', 'w4']
keyword_cols = ['k' + str(x) for x in range(1, 16)]
col_sets = [sentiment_cols, time_cols, keyword_cols]
models = []
predictions = []

for cols in col_sets:
    clf = linear_model.RidgeCV(alphas = np.linspace(0, 5, 10), cv = 5, normalize = True)
    y = np.array(train_set[cols])
    clf.fit(X, y)
    pred = clf.predict(Xp)
    print 'Best Alpha:', clf.alpha_
    print 'Train error: {0}'.format(np.sqrt(np.sum(np.array(np.array(clf.predict(X))-y)**2)/ (X.shape[0]*24.0)))
    predictions.append(pred)
    models.append(clf)

# Get rid of negative numbers
for pred in predictions:
    pred[pred < 0] = 0

# Normalize the sets of tags that must sum to 1
sentiment_preds = predictions[0]
time_preds = predictions[1]
keyword_preds = predictions[2]

# Normalize sentiment
rowsums = sentiment_preds.sum(1).reshape(-1, 1)
sentiment_preds = sentiment_preds / np.tile(rowsums, (1, 5))

# Normalize time
rowsums = time_preds.sum(1).reshape(-1, 1)
time_preds = time_preds / np.tile(rowsums, (1, 4))

txt(f, prediction, col, delimiter=',')
all_predictions = np.hstack([sentiment_preds, time_preds, keyword_preds])
output('results/split_ridge.csv', all_predictions)
