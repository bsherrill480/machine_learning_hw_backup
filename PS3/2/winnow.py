import util
import numpy as np

def train_winnow(training_data, num_rounds, learning_rate, estimated_margin, threshold,
                 S=None, S_timeout=util.fake_infinity):
    dimensionality = len(training_data[0].features)
    # weights = [1 for i in xrange(dimensionality)]  # w <- 0
    weights = np.ones(dimensionality, dtype=float)
    curr_round = 0  # t = 0
    num_mistakes = 0
    running_rounds_without_mistake = 0
    while curr_round < num_rounds and curr_round < S_timeout:  # while t < T
        for training_datum in training_data:  # for all (x, y) in S
            label = training_datum.label
            features = training_datum.np_features
            prediction = label * (np.dot(weights, features) - threshold)  # y(w * x + threshold)
            if prediction <= estimated_margin:  # y(w * x - theta) <= estimated_margin
                # w <- w o n^(yx)
                # weights = [w * (learning_rate ** (label * features[i])) for i, w in enumerate(weights)]
                learning_rate_array = np.full(dimensionality, learning_rate, dtype=float)
                weights = np.multiply(
                    weights,
                    learning_rate_array ** (features * label)
                )
                num_mistakes += 1
                running_rounds_without_mistake = 0
        curr_round += 1  # t <- t + 1
        if S and running_rounds_without_mistake >= S:
            return weights, threshold, num_mistakes

    if curr_round == S_timeout:
        print 'winnow S timeout'

    return weights, threshold, num_mistakes

# keeping around as backup
def train_winnow_no_numpy(training_data, num_rounds, learning_rate, estimated_margin, threshold,
                 S=None, S_timeout=util.fake_infinity):
    dimensionality = len(training_data[0].features)
    weights = [1 for i in xrange(dimensionality)]  # w <- 0
    curr_round = 0  # t = 0
    num_mistakes = 0
    running_rounds_without_mistake = 0
    while curr_round < num_rounds and curr_round < S_timeout:  # while t < T
        for training_datum in training_data:  # for all (x, y) in S
            label = training_datum.label
            features = training_datum.features
            prediction = label * (util.dot(weights, features) - threshold)  # y(w * x + threshold)
            if prediction <= estimated_margin:  # y(w * x - theta) <= estimated_margin
                # w <- w o n^(yx)
                weights = [w * (learning_rate ** (label * features[i])) for i, w in enumerate(weights)]
                num_mistakes += 1
                running_rounds_without_mistake = 0
        curr_round += 1  # t <- t + 1
        if S and running_rounds_without_mistake >= S:
            return weights, threshold, num_mistakes

    if curr_round == S_timeout:
        print 'winnow S timeout'

    return weights, threshold, num_mistakes


def classify_winnow(weights, theta, features):
    activation = util.dot(weights, features) - theta
    return -1 if activation < 0 else 1

