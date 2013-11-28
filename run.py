from load import *
from clean import *
from model import *
import os

test_set = load_test()
train_set = load_train()


def output(filename, predictions):
    prediction = np.array(np.hstack([np.matrix(test_set['id']).T, predictions]))
    col = '%i,' + '%f,'*23 + '%f'
    header = ','.join(['id'] + list(train_set.columns[4:])) + '\n'

    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, 'w') as f:
        f.write(header)
        np.savetxt(f, prediction, col, delimiter=',')


def ridge_001():
    print('*** CLEANING ***')
    tfidf = get_tfidf_vectorizer(train_set['tweet'])
    X_train = tfidf.transform(train_set['tweet'])
    X_test = tfidf.transform(test_set['tweet'])
    y_train = np.array(train_set.ix[:, 4:])

    print('*** TRAINING ***')
    mdl = ridge(X_train, y_train)
    print('*** PREDICTING ***')
    test_prediction = mdl.predict(X_test)
    print('*** OUTPUTTING ***')
    output('results/ridge_001.csv', test_prediction)


def ridge_002():
    print('*** CLEANING ***')
    tfidf = get_tfidf_vectorizer(train_set['tweet'])
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
        mdl = ridge(X_train, y)
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

    sentiment_preds = normalize_sum_to_one(sentiment_preds)
    time_preds = normalize_sum_to_one(time_preds)

    all_predictions = np.hstack([sentiment_preds, time_preds, keyword_preds])
    print('*** OUTPUTTING ***')
    output('results/ridge_002.csv', all_predictions)
