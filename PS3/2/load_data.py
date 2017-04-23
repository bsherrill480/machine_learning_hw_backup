import os
import math
import numpy as np


def _get_file_path_for_material(file_name):
    return '../materials/{}'.format(file_name)


def _get_lines_from_file(file_name):
    relative_file_location = _get_file_path_for_material(file_name)
    absolute_file_location = os.path.join(os.path.dirname(__file__), relative_file_location)
    lines = tuple(open(absolute_file_location, 'r'))
    return lines


class LabeledFeature:
    def __init__(self, label, features, np_features):
        self.label = label
        self.features = features
        self.np_features = np_features


def _make_feature(file_row, dimensionality):
    feature_full_row = [-1 for i in range(dimensionality)]
    existing_features = file_row.split(' ')
    label = int(existing_features.pop(0))  # remove the label
    for existing_feature in existing_features:
        feature, exists = existing_feature.split(':')
        feature = int(feature) - 1  # we're using 0 based indexing, not 1
        feature_full_row[feature] = 1
    return LabeledFeature(label, feature_full_row, np.array(feature_full_row, dtype=float))


def get_labeled_data(file_name, dimensionality):
    return [_make_feature(file_row, dimensionality) for file_row in _get_lines_from_file(file_name)]
