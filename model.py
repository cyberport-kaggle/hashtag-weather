import numpy as np
from sklearn import linear_model


def rmse(model, data, actuals):
    return np.sqrt(np.sum(np.array(np.array(model.predict(data)) - actuals) ** 2) / (data.shape[0] * 24.0))


def ridge(data, actuals):
    clf = linear_model.RidgeCV(alphas=np.linspace(0, 5, 10), cv=5, normalize=True)
    clf.fit(data, actuals)
    print 'Best Alpha:', clf.alpha_
    print 'Train error: {0}'.format(rmse(clf, data, actuals))
    return clf


def grouped_ridge(data, actuals):
    pass
