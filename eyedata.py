import pandas as pd
import numpy as np


def get_raw_tests():
    data = pd.read_csv('eyedata.csv')

    all_tests = []
    prev_key = None
    curr_test = []

    for row in data.iterrows():
        _, cols = row

        subj, block, ttype, *_ = cols

        key = subj, block, ttype
        if key != prev_key:
            if prev_key is not None:
                all_tests.append(pd.DataFrame(curr_test))
                curr_test = []

            prev_key = key

        curr_test.append(cols)
    return all_tests


def convert_test_to_training_data(test: pd.DataFrame):
    np_df = test.to_numpy()
    # indices = subj, block, ttype, mag, amp, dur
    x = np_df[:, [3, 4, 5]]
    y = test['ttype'].iloc[0]
    y = np.eye(3)[['DIFF', 'EASY', 'CONTROL'].index(y)]
    return x.astype(np.float64), y.astype(np.int32)


def get_training_data():
    X = []
    Y = []

    for test in get_raw_tests():
        x, y = convert_test_to_training_data(test)
        if len(x) > 100:
            X.append(x[-100:].T)
            Y.append(y)
        # X.append(x)
        # Y.append(y)

    return np.array(X), np.array(Y)
