import util
import numpy as np

def train_perceptron(training_data, num_rounds, learning_rate, estimated_marin, S=None,
                     S_timeout=util.fake_infinity):
    dimensionality = len(training_data[0].features)
    weights = np.zeros(dimensionality, dtype=float)  # w <- 0
    curr_round = 0  # t = 0
    num_mistakes = 0
    b = 0
    running_rounds_without_mistake = 0
    while curr_round < num_rounds and curr_round < S_timeout:  # while t < T
        for training_datum in training_data:  # for all (x, y) in S
            label = training_datum.label
            features = training_datum.np_features
            prediction = label * (np.dot(weights, features) + b)  # y(w * x + threshold)
            is_mistake = prediction <= estimated_marin  # y(w * x + threshold) <= estimated_margin
            if is_mistake:
                # w <- w + learning_rate * y * x
                weights = np.add(
                    weights,
                    features * label * learning_rate
                )
                b = b + learning_rate * label  # threshold <- threshold + learning_rate * y
                num_mistakes += 1
                running_rounds_without_mistake = 0
        curr_round += 1  # t <- t + 1
        if S and running_rounds_without_mistake >= S:
            return weights, b, num_mistakes

    if curr_round == S_timeout:
        print 'perceptron S timeout'

    return weights, b, num_mistakes


# keeping around as backup
def train_perceptron_no_numpy(training_data, num_rounds, learning_rate, estimated_marin, S=None,
                     S_timeout=util.fake_infinity):
    dimensionality = len(training_data[0].features)
    weights = [0 for i in xrange(dimensionality)]  # w <- 0
    curr_round = 0  # t = 0
    num_mistakes = 0
    b = 0
    running_rounds_without_mistake = 0
    while curr_round < num_rounds and curr_round < S_timeout:  # while t < T
        for training_datum in training_data:  # for all (x, y) in S
            label = training_datum.label
            features = training_datum.features
            prediction = label * (util.dot(weights, features) + b)  # y(w * x + threshold)
            is_mistake = prediction <= estimated_marin  # y(w * x + threshold) <= estimated_margin
            if is_mistake:
                # w <- w + learning_rate * y * x
                weights = [w + learning_rate * label * features[i] for i, w in enumerate(weights)]
                b = b + learning_rate * label  # threshold <- threshold + learning_rate * y
                num_mistakes += 1
                running_rounds_without_mistake = 0
        curr_round += 1  # t <- t + 1
        if S and running_rounds_without_mistake >= S:
            return weights, b, num_mistakes

    if curr_round == S_timeout:
        print 'perceptron S timeout'

    return weights, b, num_mistakes

def classify_perceptron(weights, b, features):
    activation = util.dot(weights, features) + b
    return -1 if activation < 0 else 1

