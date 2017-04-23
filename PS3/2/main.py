import load_data
import perceptron
import winnow
import util
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager
from time import sleep

NUM_ROUNDS_A = 20
TRAINING_DATA_FRACTION = 0.1
PERCEPTRON = 'perceptron'
WINNOW = 'winnnow'

ESTIMATED_MARGINS = [0, 1]  # if is winnow and estimated_margin == 1, then do greater than 0 array
ESTIMATED_MARGINS_WINNOW_GREATER_THAN_0 = [2.0, 0.3, 0.04, 0.006, 0.001]
# e.g. LEARNING_RATES[algorithm][estimated_margin]
LEARNING_RATES = {
    PERCEPTRON: {
        '0': [1],
        '1': [1.5, 0.25, 0.03, 0.005, 0.001],
    },
    WINNOW: [1.1, 1.01, 1.005, 1.0005, 1.0001]
}
# part B
# Increasing these to the numbers to in 1000 and 10000 resulted in a small gain, ~1%,
# but increased running time by a lot.
S = 100
S_TIMEOUT = 500

# part C
NUM_ROUNDS_C = 20

class Trial(object):
    def __init__(self, train_algo_data):
        self.train_algo_data = train_algo_data

    def run(self):
        raise NotImplementedError()

    def predict(self, features):
        raise NotImplementedError()

    def get_accuracy(self, testing_data):
        self.run()
        num_right = sum(
            [1 if self.predict(testing_datum.features) == testing_datum.label else 0
             for testing_datum in testing_data]
        )
        return float(num_right) / len(testing_data)


class PerceptronTrial(Trial):
    def run(self):
        self.weights, self.b, self.num_mistakes = perceptron.train_perceptron(
            self.train_algo_data,
            self.num_rounds,
            self.learning_rate,
            self.estimated_margin,
            **self.kwargs_for_running
        )

    def predict(self, features):
        return perceptron.classify_perceptron(self.weights, self.b, features)

    def __init__(self, train_algo_data, learning_rate, estimated_margin, num_rounds, **kwargs):
        super(PerceptronTrial, self).__init__(train_algo_data)
        self.learning_rate = learning_rate
        self.estimated_margin = estimated_margin
        self.kwargs_for_running = kwargs
        self.num_rounds = num_rounds

    def __repr__(self):
        return 'learning_rate: {}, estimated_margin: {}'.format(self.learning_rate, self.estimated_margin)

class WinnowTrial(Trial):
    def run(self):
        self.weights, self.threshold, self.num_mistakes = winnow.train_winnow(
            self.train_algo_data,
            self.num_rounds,
            self.learning_rate,
            self.estimated_margin,
            500,
            **self.kwargs_for_running
        )

    def predict(self, features):
        return winnow.classify_winnow(self.weights, self.threshold, features)

    def __init__(self, train_algo_data, learning_rate, estimated_margin, num_rounds, **kwargs):
        super(WinnowTrial, self).__init__(train_algo_data)
        self.learning_rate = learning_rate
        self.estimated_margin = estimated_margin
        self.kwargs_for_running = kwargs
        self.num_rounds = num_rounds

    def __repr__(self):
        return 'learning_rate: {}, estimated_margin: {}'.format(self.learning_rate, self.estimated_margin)


class TrialResult(object):
    def __init__(self, trial, accuracy):
        self.trail = trial
        self.accuracy = accuracy

    def __repr__(self):
        return 'D2 accuracy: {}, trial: {}'.format(self.accuracy, self.trail)


def run_perceptron(result_list, train_data, learning_rate, estimated_margin,
                   hyper_param_testing_data, num_rounds, **kwargs):
    perceptron_trial = PerceptronTrial(train_data, learning_rate, estimated_margin, num_rounds,
                                       **kwargs)
    accuracy = perceptron_trial.get_accuracy(hyper_param_testing_data)
    result_list.append(TrialResult(perceptron_trial, accuracy))


def run_winnow(result_list, train_data, learning_rate, estimated_margin,
               hyper_param_testing_data, num_rounds, **kwargs):
    winnow_trial = WinnowTrial(train_data, learning_rate, estimated_margin, num_rounds, **kwargs)
    accuracy = winnow_trial.get_accuracy(hyper_param_testing_data)
    result_list.append(TrialResult(winnow_trial, accuracy))


def make_table_dict_key(n, algo, estimated_margin, key):
    return '{}/{}/{}/{}'.format(n, algo, estimated_margin, key)

def problem_a():
    DATA = {
        'a.10.100.500.txt': load_data.get_labeled_data('a.10.100.500.txt', 500),
        'a.10.100.1000.txt': load_data.get_labeled_data('a.10.100.1000.txt', 1000)
    }
    n_set = [500, 1000]
    manager = Manager()
    table_dict = manager.dict()
    for n in n_set:
        for estimated_margin in ESTIMATED_MARGINS:
            table_dict[make_table_dict_key(n, PERCEPTRON, estimated_margin, 'result')] = None
            table_dict[make_table_dict_key(n, PERCEPTRON, estimated_margin, 'M')] = None
            table_dict[make_table_dict_key(n, WINNOW, estimated_margin, 'result')] = None
            table_dict[make_table_dict_key(n, WINNOW, estimated_margin, 'M')] = None

    for n in n_set:
        # setup data
        data = DATA['a.10.100.{}.txt'.format(n)]
        data_len = len(data)
        training_data_fraction_num = int(TRAINING_DATA_FRACTION * data_len)
        train_data = data[0:training_data_fraction_num]  # 10 percent
        hyper_param_testing_data = data[training_data_fraction_num:2*training_data_fraction_num]
        graph_data = {WINNOW: dict(), PERCEPTRON: dict()}

        # setup algo
        for estimated_margin in ESTIMATED_MARGINS:
            learning_rate_perceptron_result_list = manager.list()
            learning_rate_perceptron_processes = [
                Process(
                    target=run_perceptron,
                    args=(
                        learning_rate_perceptron_result_list,
                        train_data,
                        learning_rate,
                        estimated_margin,
                        hyper_param_testing_data,
                        NUM_ROUNDS_A
                    )
                ) for learning_rate in LEARNING_RATES[PERCEPTRON][str(estimated_margin)]
                ]
            learning_rate_winnow_result_list = manager.list()
            learning_rate_winnow_processes = []
            if estimated_margin == 0:
                learning_rate_winnow_processes = [
                    Process(
                        target=run_winnow,
                        args=(
                            learning_rate_winnow_result_list,
                            train_data,
                            learning_rate,
                            estimated_margin,
                            hyper_param_testing_data,
                            NUM_ROUNDS_A
                        )
                    ) for learning_rate in LEARNING_RATES[WINNOW]
                    ]
            else:
                for winnow_estimated_margin in ESTIMATED_MARGINS_WINNOW_GREATER_THAN_0:
                    for learning_rate in LEARNING_RATES[WINNOW]:
                        learning_rate_winnow_processes.append(
                            Process(
                                target=run_winnow,
                                args=(
                                    learning_rate_winnow_result_list,
                                    train_data,
                                    learning_rate,
                                    winnow_estimated_margin,
                                    hyper_param_testing_data,
                                    NUM_ROUNDS_A
                                )
                            )
                        )
            for process in learning_rate_perceptron_processes + learning_rate_winnow_processes:
                process.start()
            for process in learning_rate_perceptron_processes + learning_rate_winnow_processes:
                process.join()
            best_perceptron_run = max(learning_rate_perceptron_result_list, key=lambda x: x.accuracy)
            best_perceptron = best_perceptron_run.trail

            best_winnow_run = max(learning_rate_winnow_result_list, key=lambda x: x.accuracy)
            best_winnow = best_winnow_run.trail

            # calcualte stuff for graph and table for percep
            num_mistakes = 0
            percp_mistake_points = []
            for i, datum in enumerate(data):
                label = datum.label
                features = datum.features
                prediction = perceptron.classify_perceptron(best_perceptron.weights,
                                                            best_perceptron.b,
                                                            features)
                if prediction != label:
                    num_mistakes += 1
                    percp_mistake_points.append((i, num_mistakes,))

            table_dict[make_table_dict_key(n, PERCEPTRON, estimated_margin, 'result')] = \
                best_perceptron_run
            table_dict[make_table_dict_key(n, PERCEPTRON, estimated_margin, 'M')] = num_mistakes
            num_mistakes = 0
            winnow_mistake_points = []
            for i, datum in enumerate(data):
                label = datum.label
                features = datum.features
                prediction = winnow.classify_winnow(best_winnow.weights,
                                                    best_winnow.threshold,
                                                    features,
                                                    )
                if prediction != label:
                    num_mistakes += 1
                    winnow_mistake_points.append((i, num_mistakes,))

            table_dict[make_table_dict_key(n, WINNOW, estimated_margin, 'result')] = \
                best_winnow_run
            table_dict[make_table_dict_key(n, WINNOW, estimated_margin, 'M')] = num_mistakes
            graph_data[PERCEPTRON][str(estimated_margin)] = percp_mistake_points
            graph_data[WINNOW][str(estimated_margin)] = winnow_mistake_points

        # generate graph
        plt.clf()
        for estimated_margin in ESTIMATED_MARGINS:
            percep_points = graph_data[PERCEPTRON][str(estimated_margin)]
            winnow_points = graph_data[WINNOW][str(estimated_margin)]
            plt.plot(
                [p[0] for p in percep_points],
                [p[1] for p in percep_points],
                label='perceptron n={} estimated_margin={}'.format(n, estimated_margin)
            )
            plt.plot(
                [p[0] for p in winnow_points],
                [p[1] for p in winnow_points],
                label='winnow n={} estimated_margin={}'.format(n, estimated_margin)
            )
        plt.legend()
        plt.savefig(util.get_file_path('a_output', 'n_{}'.format(n)))

    util.save_table_dict('a_output', 'table.txt', table_dict)

def problem_b_run_nset(n, manager, table_dict, DATA):
    # setup data
    data = DATA['b.10.20.{}.txt'.format(n)]
    data_len = len(data)
    training_data_fraction_num = int(TRAINING_DATA_FRACTION * data_len)
    train_data = data[0:training_data_fraction_num]  # 10 percent
    hyper_param_testing_data = data[training_data_fraction_num:2*training_data_fraction_num]
    graph_data = {WINNOW: dict(), PERCEPTRON: dict()}

    # setup algo
    algo_kwargs = {'S': S, 'S_timeout': S_TIMEOUT}
    for estimated_margin in ESTIMATED_MARGINS:
        learning_rate_perceptron_result_list = manager.list()
        learning_rate_perceptron_processes = [
            Process(
                target=run_perceptron,
                args=(
                    learning_rate_perceptron_result_list,
                    train_data,
                    learning_rate,
                    estimated_margin,
                    hyper_param_testing_data,
                    util.fake_infinity
                ),
                kwargs=algo_kwargs
            ) for learning_rate in LEARNING_RATES[PERCEPTRON][str(estimated_margin)]
            ]
        learning_rate_winnow_result_list = manager.list()
        learning_rate_winnow_processes = []
        if estimated_margin == 0:
            learning_rate_winnow_processes = [
                Process(
                    target=run_winnow,
                    args=(
                        learning_rate_winnow_result_list,
                        train_data,
                        learning_rate,
                        estimated_margin,
                        hyper_param_testing_data,
                        util.fake_infinity
                    ),
                    kwargs=algo_kwargs
                ) for learning_rate in LEARNING_RATES[WINNOW]
                ]
        else:
            for winnow_estimated_margin in ESTIMATED_MARGINS_WINNOW_GREATER_THAN_0:
                for learning_rate in LEARNING_RATES[WINNOW]:
                    learning_rate_winnow_processes.append(
                        Process(
                            target=run_winnow,
                            args=(
                                learning_rate_winnow_result_list,
                                train_data,
                                learning_rate,
                                winnow_estimated_margin,
                                hyper_param_testing_data,
                                util.fake_infinity
                            ),
                            kwargs=algo_kwargs
                        )
                    )
        for process in learning_rate_perceptron_processes + learning_rate_winnow_processes:
            process.start()
        for process in learning_rate_perceptron_processes + learning_rate_winnow_processes:
            process.join()
        best_perceptron_run = max(learning_rate_perceptron_result_list, key=lambda x: x.accuracy)
        best_perceptron = best_perceptron_run.trail

        best_winnow_run = max(learning_rate_winnow_result_list, key=lambda x: x.accuracy)
        best_winnow = best_winnow_run.trail

        # calcualte stuff for graph and table for percep
        num_mistakes = 0
        percp_mistake_points = []
        for i, datum in enumerate(data):
            label = datum.label
            features = datum.features
            prediction = perceptron.classify_perceptron(best_perceptron.weights,
                                                        best_perceptron.b,
                                                        features)
            if prediction != label:
                num_mistakes += 1
                percp_mistake_points.append((i, num_mistakes,))

        table_dict[make_table_dict_key(n, PERCEPTRON, estimated_margin, 'result')] = \
            best_perceptron_run
        table_dict[make_table_dict_key(n, PERCEPTRON, estimated_margin, 'M')] = num_mistakes
        num_mistakes = 0
        winnow_mistake_points = []
        for i, datum in enumerate(data):
            label = datum.label
            features = datum.features
            prediction = winnow.classify_winnow(best_winnow.weights,
                                                best_winnow.threshold,
                                                features,
                                                )
            if prediction != label:
                num_mistakes += 1
                winnow_mistake_points.append((i, num_mistakes,))

        table_dict[make_table_dict_key(n, WINNOW, estimated_margin, 'result')] = \
            best_winnow_run
        table_dict[make_table_dict_key(n, WINNOW, estimated_margin, 'M')] = num_mistakes
        graph_data[PERCEPTRON][str(estimated_margin)] = percp_mistake_points
        graph_data[WINNOW][str(estimated_margin)] = winnow_mistake_points
        print('done nset {} estimated margin: {}'.format(n, estimated_margin))
    plt.clf()
    for estimated_margin in ESTIMATED_MARGINS:
        percep_points = graph_data[PERCEPTRON][str(estimated_margin)]
        winnow_points = graph_data[WINNOW][str(estimated_margin)]
        plt.plot(
            [p[0] for p in percep_points],
            [p[1] for p in percep_points],
            label='perceptron n={} estimated_margin={}'.format(n, estimated_margin)
        )
        plt.plot(
            [p[0] for p in winnow_points],
            [p[1] for p in winnow_points],
            label='winnow n={} estimated_margin={}'.format(n, estimated_margin)
        )
    plt.legend()
    plt.savefig(util.get_file_path('b_output', 'n_{}'.format(n)))

    print('done nset {}'.format(n))

def problem_b():
    manager = Manager()
    DATA = {
        'b.10.20.40.txt': load_data.get_labeled_data('b.10.20.40.txt', 40),
        'b.10.20.80.txt': load_data.get_labeled_data('b.10.20.80.txt', 80),
        'b.10.20.120.txt': load_data.get_labeled_data('b.10.20.120.txt', 120),
        'b.10.20.160.txt': load_data.get_labeled_data('b.10.20.160.txt', 160),
        'b.10.20.200.txt': load_data.get_labeled_data('b.10.20.200.txt', 200),
    }
    n_set = [40, 80, 120, 160, 200]
    table_dict = manager.dict()
    for n in n_set:
        for estimated_margin in ESTIMATED_MARGINS:
            table_dict[make_table_dict_key(n, PERCEPTRON, estimated_margin, 'result')] = None
            table_dict[make_table_dict_key(n, PERCEPTRON, estimated_margin, 'M')] = None
            table_dict[make_table_dict_key(n, WINNOW, estimated_margin, 'result')] = None
            table_dict[make_table_dict_key(n, WINNOW, estimated_margin, 'M')] = None

    n_set_processes = [
        Process(
            target=problem_b_run_nset,
            args=(n, manager, table_dict, DATA,)
        )
        for n in n_set
        ]
    for p in n_set_processes:
        p.start()
    for p in n_set_processes:
        p.join()

    #
    print('print table')
    util.save_table_dict('b_output', 'table.txt', table_dict)


def problem_c_run_mset(m, manager, table_dict, DATA):
    print('starting problem_c_run_mset {}'.format(m))
    # setup data
    data = DATA['c.10.{}.1000.train.txt'.format(m)]
    data_test = DATA['c.10.{}.1000.test.txt'.format(m)]
    data_len = len(data)

    training_data_fraction_num = int(TRAINING_DATA_FRACTION * data_len)
    train_data = data[0:training_data_fraction_num]  # 10 percent
    hyper_param_testing_data = data[training_data_fraction_num:2*training_data_fraction_num]


    # data_len_split = int(data_len / 2)
    # train_data = manager.list(data[0:data_len_split])
    # hyper_param_testing_data = manager.list(data[data_len_split:data_len])



    # graph_data = {WINNOW: dict(), PERCEPTRON: dict()}

    # setup algo
    for estimated_margin in ESTIMATED_MARGINS:
        print('m:{} estimated_margin: {}'.format(m, estimated_margin))
        learning_rate_perceptron_result_list = manager.list()
        learning_rate_perceptron_processes = [
            Process(
                target=run_perceptron,
                args=(
                    learning_rate_perceptron_result_list,
                    train_data,
                    learning_rate,
                    estimated_margin,
                    hyper_param_testing_data,
                    NUM_ROUNDS_C
                ),
            ) for learning_rate in LEARNING_RATES[PERCEPTRON][str(estimated_margin)]
            ]
        learning_rate_winnow_result_list = manager.list()
        learning_rate_winnow_processes = []
        if estimated_margin == 0:
            learning_rate_winnow_processes = [
                Process(
                    target=run_winnow,
                    args=(
                        learning_rate_winnow_result_list,
                        train_data,
                        learning_rate,
                        estimated_margin,
                        hyper_param_testing_data,
                        NUM_ROUNDS_C
                    ),
                ) for learning_rate in LEARNING_RATES[WINNOW]
                ]
        else:
            for winnow_estimated_margin in ESTIMATED_MARGINS_WINNOW_GREATER_THAN_0:
                for learning_rate in LEARNING_RATES[WINNOW]:
                    learning_rate_winnow_processes.append(
                        Process(
                            target=run_winnow,
                            args=(
                                learning_rate_winnow_result_list,
                                train_data,
                                learning_rate,
                                winnow_estimated_margin,
                                hyper_param_testing_data,
                                NUM_ROUNDS_C
                            ),
                        )
                    )
        # DUE TO MEMORY CONSTRAINS WE'LL DO THIS 1 AT A TIME
        for process in learning_rate_perceptron_processes + learning_rate_winnow_processes:
            print('startin process')
            process.start()
            process.join()
            print('joining process')
        print('done with processes')
        # for process in learning_rate_perceptron_processes + learning_rate_winnow_processes:
        #     process.start()
        # for process in learning_rate_perceptron_processes + learning_rate_winnow_processes:
        #     process.join()
        best_perceptron_run = max(learning_rate_perceptron_result_list, key=lambda x: x.accuracy)
        best_perceptron = best_perceptron_run.trail

        best_winnow_run = max(learning_rate_winnow_result_list, key=lambda x: x.accuracy)
        best_winnow = best_winnow_run.trail

        # calcualte stuff for graph and table for percep
        num_mistakes = 0
        percp_mistake_points = []
        for i, datum in enumerate(data_test):
            label = datum.label
            features = datum.features
            prediction = perceptron.classify_perceptron(best_perceptron.weights,
                                                        best_perceptron.b,
                                                        features)
            if prediction != label:
                num_mistakes += 1
                percp_mistake_points.append((i, num_mistakes,))

        table_dict[make_table_dict_key(m, PERCEPTRON, estimated_margin, 'result')] = \
            best_perceptron_run
        table_dict[make_table_dict_key(m, PERCEPTRON, estimated_margin, 'M')] = num_mistakes
        num_mistakes = 0
        winnow_mistake_points = []
        for i, datum in enumerate(data_test):
            label = datum.label
            features = datum.features
            prediction = winnow.classify_winnow(best_winnow.weights,
                                                best_winnow.threshold,
                                                features,
                                                )
            if prediction != label:
                num_mistakes += 1
                winnow_mistake_points.append((i, num_mistakes,))

        table_dict[make_table_dict_key(m, WINNOW, estimated_margin, 'result')] = \
            best_winnow_run
        table_dict[make_table_dict_key(m, WINNOW, estimated_margin, 'M')] = num_mistakes
        # graph_data[PERCEPTRON][str(estimated_margin)] = percp_mistake_points
        # graph_data[WINNOW][str(estimated_margin)] = winnow_mistake_points
        print('done nset {} estimated margin: {}'.format(m, estimated_margin))
    # plt.clf()
    # for estimated_margin in ESTIMATED_MARGINS:
    #     percep_points = graph_data[PERCEPTRON][str(estimated_margin)]
    #     winnow_points = graph_data[WINNOW][str(estimated_margin)]
    #     plt.plot(
    #         [p[0] for p in percep_points],
    #         [p[1] for p in percep_points],
    #         label='perceptron m={} estimated_margin={}'.format(m, estimated_margin)
    #     )
    #     plt.plot(
    #         [p[0] for p in winnow_points],
    #         [p[1] for p in winnow_points],
    #         label='winnow m={} estimated_margin={}'.format(m, estimated_margin)
    #     )
    # plt.legend()
    # plt.savefig(util.get_file_path('b_output', 'n_{}'.format(m)))

    print('done mset {}'.format(m))

def problem_c():
    manager = Manager()
    table_dict = manager.dict()
    m = 1000
    DATA = manager.dict({
        # 'c.10.100.1000.train.txt': load_data.get_labeled_data('c.10.100.1000.train.txt', 1000),
        # 'c.10.100.1000.test.txt': load_data.get_labeled_data('c.10.100.1000.test.txt', 1000),

        'c.10.{}.1000.train.txt'.format(m): load_data.get_labeled_data(
            'c.10.{}.1000.train.txt'.format(m), 1000
        ),
        'c.10.{}.1000.test.txt'.format(m): load_data.get_labeled_data(
            'c.10.{}.1000.test.txt'.format(m), 1000
        ),
    })
    print('files loaded')

    #code for if we wanted to run run parallel, but due to memory we can't

    # m_set = [500]
    # for m in m_set:
    #     for estimated_margin in ESTIMATED_MARGINS:
    #         table_dict[make_table_dict_key(m, PERCEPTRON, estimated_margin, 'result')] = None
    #         table_dict[make_table_dict_key(m, PERCEPTRON, estimated_margin, 'M')] = None
    #         table_dict[make_table_dict_key(m, WINNOW, estimated_margin, 'result')] = None
    #         table_dict[make_table_dict_key(m, WINNOW, estimated_margin, 'M')] = None
    #
    # m_set_processes = [
    #     Process(
    #         target=problem_c_run_mset,
    #         args=(m, manager, table_dict, DATA,)
    #     )
    #     for m in m_set
    #     ]
    # for p in m_set_processes:
    #     p.start()
    # for p in m_set_processes:
    #     p.join()
    #
    # #
    # print('print table')
    # util.save_table_dict('c_output', 'm_{}_table.txt', table_dict, m_or_n='m')

    # Run

    for estimated_margin in ESTIMATED_MARGINS:
        table_dict[make_table_dict_key(m, PERCEPTRON, estimated_margin, 'result')] = None
        table_dict[make_table_dict_key(m, PERCEPTRON, estimated_margin, 'M')] = None
        table_dict[make_table_dict_key(m, WINNOW, estimated_margin, 'result')] = None
        table_dict[make_table_dict_key(m, WINNOW, estimated_margin, 'M')] = None

    problem_c_run_mset(m, manager, table_dict, DATA)

    util.save_table_dict('c_output', 'm_{}_table.txt'.format(m), table_dict, m_or_n='m')

if __name__ == "__main__":
    # problem_a()
    # problem_b()
    problem_c()



