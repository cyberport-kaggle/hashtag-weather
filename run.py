import load, clean, model
import numpy as np
import sklearn
import os
from IPython import embed
from sklearn.feature_extraction.text import TfidfVectorizer

test_set = load.load_test()
train_set = load.load_train()


def output(filename, predictions):
    prediction = np.array(np.hstack([np.matrix(test_set['id']).T, predictions]))
    col = '%i,' + '%f,'*23 + '%f'
    header = ','.join(['id'] + list(train_set.columns[4:])) + '\n'

    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, 'w') as f:
        f.write(header)
        np.savetxt(f, prediction, col, delimiter=',')


def ridge_001(column_errs=False):
    print('*** CLEANING ***')
    tfidf = clean.get_tfidf_vectorizer(train_set['tweet'])
    X_train = tfidf.transform(train_set['tweet'])
    X_test = tfidf.transform(test_set['tweet'])
    y_train = np.array(train_set.ix[:, 4:])
    print('*** TRAINING ***')
    mdl = model.ridge(X_train, y_train)
    final_mdl = sklearn.linear_model.Ridge(alpha=mdl.alpha_, normalize=True)
    cv_error = model.cv(final_mdl, X_train, y_train)
    print "CV Errors: {}, Mean: {}".format(cv_error, sum(cv_error) / len(cv_error))

    if column_errs:
        columns = list(train_set.columns[4:])
        final_mdl.fit(X_train, y_train)
        predictions = final_mdl.predict(X_train)
        rmses = []
        for i, col in enumerate(columns):
            err = np.sqrt(np.sum(np.array(predictions[:, i] - y_train[:, i]) ** 2) / (X_train.shape[0]))
            print "Column {} RMSE: {}".format(col, err)
            rmses.append(err)

    print('*** PREDICTING ***')
    test_prediction = mdl.predict(X_test)
    print('*** OUTPUTTING ***')
    output('results/ridge_001.csv', test_prediction)


def ridge_002():
    print('*** CLEANING ***')
    tfidf = clean.get_tfidf_vectorizer(train_set['tweet'])
    X_train = tfidf.transform(train_set['tweet'])
    X_test = tfidf.transform(test_set['tweet'])

    # Split the regression so that we regress for each set of hashtags
    sentiment_cols = ['s1', 's2', 's3', 's4', 's5']
    time_cols = ['w1', 'w2', 'w3', 'w4']
    keyword_cols = ['k' + str(x) for x in range(1, 16)]
    col_sets = [sentiment_cols, time_cols, keyword_cols]
    models = []
    predictions = []

    for cols in col_sets:
        y = np.array(train_set[cols])
        print('*** TRAINING ***')
        mdl = model.ridge(X_train, y)
        print('*** PREDICTING ***')
        pred = mdl.predict(X_test)
        predictions.append(pred)
        models.append(mdl)

    # Get rid of negative numbers
    for pred in predictions:
        pred[pred < 0] = 0

    # Normalize the sets of tags that must sum to 1
    sentiment_preds = predictions[0]
    time_preds = predictions[1]
    keyword_preds = predictions[2]

    sentiment_preds = clean.normalize_sum_to_one(sentiment_preds)
    time_preds = clean.normalize_sum_to_one(time_preds)

    all_predictions = np.hstack([sentiment_preds, time_preds, keyword_preds])
    print('*** OUTPUTTING ***')
    output('results/ridge_002.csv', all_predictions)


def ridge_003():
    print('*** CLEANING ***')
    tfidf_wrd = TfidfVectorizer(max_features=10000, strip_accents='unicode', analyzer='word', ngram_range=(1, 3),
                                lowercase=True, stop_words='english', min_df=3, max_df=0.5)
    tfidf_wrd.fit(train_set['tweet'])
    X_train_wrd = tfidf_wrd.transform(train_set['tweet'])
    X_test_wrd = tfidf_wrd.transform(test_set['tweet'])

    tfidf_char = TfidfVectorizer(max_features=10000, strip_accents='unicode', analyzer='char', ngram_range=(4, 10),
                                lowercase=True, stop_words='english', min_df=3, max_df=0.5)
    tfidf_char.fit(train_set['tweet'])
    X_train_char = tfidf_char.transform(train_set['tweet'])
    X_test_char = tfidf_char.transform(test_set['tweet'])

    y_train = np.array(train_set.ix[:, 4:])

    print('*** TRAINING ***')
    mdl_wrd = model.ridge(X_train_wrd, y_train)
    mdl_char = model.ridge(X_train_char, y_train)

    print('*** PREDICTING ***')
    test_prediction_wrd = mdl_wrd.predict(X_test_wrd)
    test_prediction_char = mdl_char.predict(X_test_char)

    test_prediction = (test_prediction_wrd + test_prediction_char) / 2

    print('*** OUTPUTTING ***')
    output('results/ridge_003.csv', test_prediction)


def ridge_004(column_errs=False):
    # Some more rigorous cleaning
    print('*** CLEANING ***')

    for i in range(train_set['tweet'].shape[0]):
        train_set['tweet'][i] = clean.clean_string(train_set['tweet'][i])

    for i in range(test_set['tweet'].shape[0]):
        test_set['tweet'][i] = clean.clean_string(test_set['tweet'][i])

    # The cleaning and stemming takes it down to ~7000 features, from the previously more than 10k
    tfidf = clean.get_tfidf_vectorizer(train_set['tweet'])

    X_train = tfidf.transform(train_set['tweet'])
    X_test = tfidf.transform(test_set['tweet'])
    y_train = np.array(train_set.ix[:, 4:])
    print('*** TRAINING ***')
    mdl = model.ridge(X_train, y_train)
    final_mdl = sklearn.linear_model.Ridge(alpha=mdl.alpha_, normalize=True)
    cv_error = model.cv(final_mdl, X_train, y_train)
    print "CV Errors: {}, Mean: {}".format(cv_error, sum(cv_error) / len(cv_error))

    if column_errs:
        columns = list(train_set.columns[4:])
        final_mdl.fit(X_train, y_train)
        predictions = final_mdl.predict(X_train)
        rmses = []
        for i, col in enumerate(columns):
            err = np.sqrt(np.sum(np.array(predictions[:, i] - y_train[:, i]) ** 2) / (X_train.shape[0]))
            print "Column {} RMSE: {}".format(col, err)
            rmses.append(err)

    print('*** PREDICTING ***')
    test_prediction = mdl.predict(X_test)

    test_prediction = clean.remove_negatives(test_prediction)
    test_prediction[:, 0:5] = clean.normalize_sum_to_one(test_prediction[:, 0:5])
    test_prediction[:, 5:9] = clean.normalize_sum_to_one(test_prediction[:, 5:9])

    print('*** OUTPUTTING ***')
    output('results/ridge_004.csv', test_prediction)
