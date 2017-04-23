import os
import csv
import json


def write_data_to_file(file_path, data):
    with open(file_path, 'w+') as f:
        for data_row in data:
            data_row_text = ' '.join([str(i) for i in data_row])
            f.write('{}\n'.format(data_row_text))


def write_labels_to_file(file_path, labels):
    with open(file_path, 'w+') as f:
        for label_row in labels:
            f.write('{}\n'.format(label_row))


def standardize_raw_labels(labels):
    return [str(l).strip() for l in labels]


def get_raw_label_to_int_map(raw_labels):
    uniq_labels = list(set(raw_labels))
    sorted_uniq_labels = sorted(uniq_labels)  # be deterministic
    labels_map = dict()
    for i, label in enumerate(sorted_uniq_labels):
        labels_map[label] = i
    return labels_map


def raw_labels_to_int_labels(raw_labels, raw_label_to_int_map):
    return [raw_label_to_int_map[l] for l in raw_labels]


def run(relative_file_path, output_dir, label_col, raw_label_to_int_map=None):
    file_name = relative_file_path.split('/')[-1].split('.')[0]

    # read file
    input_relative_file_location = relative_file_path
    input_absolute_file_location = os.path.join(os.path.dirname(__file__),
                                                input_relative_file_location)
    with open(input_absolute_file_location, 'rb') as csv_file:
        reader = csv.reader(csv_file)
        lines = list(reader)

    headers = lines[0]
    lines = lines[1:]

    label_index = headers.index(label_col)

    # exclude label column
    just_data = [l[0:label_index] + l[label_index+1:] for l in lines]

    # convert to floats
    just_data = [[float(feature) for feature in datum_row] for datum_row in just_data]

    # get labels
    raw_labels_for_data = standardize_raw_labels([l[label_index] for l in lines])
    if not raw_label_to_int_map:
        raw_label_to_int_map = get_raw_label_to_int_map(raw_labels_for_data)
    labels_for_data = raw_labels_to_int_labels(raw_labels_for_data, raw_label_to_int_map)

    # get data about file
    dimensionality = len(just_data[0])
    num_data = len(just_data)

    # get folder output loc
    output_relative_folder_location = output_dir
    output_absolute_folder_location = os.path.join(os.path.dirname(__file__),
                                                   output_relative_folder_location)

    # write data
    output_data_location = os.path.join(output_absolute_folder_location,
                                        '{}.data'.format(file_name))
    write_data_to_file(output_data_location, [[num_data, dimensionality]] + just_data)

    # write labels
    output_labels_location = os.path.join(output_absolute_folder_location,
                                          '{}.labels'.format(file_name))
    write_labels_to_file(output_labels_location, ['{} 1'.format(num_data)] + labels_for_data)

    # write labels key
    output_labels_key_location = os.path.join(output_absolute_folder_location,
                                          '{}.labels_key'.format(file_name))
    with open(output_labels_key_location, 'w+') as f:
        json.dump(raw_label_to_int_map, f, sort_keys=True, indent=4, separators=(',', ': '))

    return raw_label_to_int_map

if __name__ == '__main__':
    relative_file_path = './boston_housing_raw/BostonHousingNominalTrain.csv'
    output_dir = 'boston_housing_processed_train'
    raw_labels_to_int_map = run(relative_file_path, output_dir, 'MEDV')
    relative_file_path = './boston_housing_raw/BostonHousingNominalTest.csv'
    output_dir = 'boston_housing_processed_test'
    run(relative_file_path, output_dir, 'MEDV', raw_label_to_int_map=raw_labels_to_int_map)

