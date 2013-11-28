import pandas as p
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from IPython import embed
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

paths = ['data/train.csv', 'data/test.csv']
t = p.read_csv(paths[0])
t2 = p.read_csv(paths[1])
# print t #display the data

tfidf = TfidfVectorizer(max_features=10000, strip_accents='unicode', analyzer='word')
tfidf.fit(t['tweet'])
X = tfidf.transform(t['tweet'])
test = tfidf.transform(t2['tweet'])
y = np.array(t.ix[:,4:])

test_prediction = np.zeros(shape = (test.shape[0], y.shape[1]))

lsa = TruncatedSVD(n_components = 100)
lsa.fit(X)
X = lsa.transform(X)
test = lsa.transform(test)
print X.shape, test.shape

np.savetxt('lsa_train_x.csv', X, delimiter=',')
np.savetxt('lsa_train_y.csv', y, delimiter=',')
np.savetxt('lsa_test_x.csv', test, delimiter=',')

embed()

# for col in range(y.shape[1]):
# 	print "Training column", col
# 	clf = SVR(kernel = 'rbf', C=1e3, gamma=0.1, cache_size=1000, verbose=True)
# 	clf.fit(X, y[:, col])
# 	test_prediction[:,col] = clf.predict(test)

clf = RandomForestClassifier(verbose = 1)
clf.fit(X, y)
test_prediction = clf.predict(test)

prediction = np.array(np.hstack([np.matrix(t2['id']).T, test_prediction]))
col = '%i,' + '%f,'*23 + '%f'
header = ','.join(['id'] + list(t.columns[4:])) + '\n'

outfile = 'results/svr.csv'

if os.path.exists(outfile):
    os.remove(outfile)

with open(outfile, 'w') as f:
    f.write(header)
    np.savetxt(f, prediction, col, delimiter=',')