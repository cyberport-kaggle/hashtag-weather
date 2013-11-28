import pandas as p


def load_train():
    res = p.read_csv('data/train.csv')
    return res


def load_test():
    res = p.read_csv('data/test.csv')
    return res