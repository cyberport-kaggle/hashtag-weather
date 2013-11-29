import numpy as np
from sklearn import linear_model, cross_validation


def rmse(model, data, actuals):
    return np.sqrt(np.sum(np.array(np.array(model.predict(data)) - actuals) ** 2) / (data.shape[0] * 24.0))


def cv(model, data, actuals, folds=5):
    kf = cross_validation.KFold(data.shape[0], n_folds=folds, indices=True)
    errors = []
    for train, test in kf:
        X_train, X_test, y_train, y_test = data[train], data[test], actuals[train], actuals[test]
        model.fit(X_train, y_train)
        errors.append(rmse(model, X_test, y_test))
    return errors


def ridge(data, actuals):
    clf = linear_model.RidgeCV(alphas=np.linspace(0, 5, 10), cv=5, normalize=True)
    if data is not None and actuals is not None:
        clf.fit(data, actuals)
        print 'Best Alpha:', clf.alpha_
        print 'Train error: {0}'.format(rmse(clf, data, actuals))
    return clf


def grouped_ridge(data, actuals):
    pass
