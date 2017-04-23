import matplotlib.pyplot as plt
import os
import sys
import numpy as np

fake_infinity = sys.maxint - 1

def get_file_path(file_dir, file_name):
    relative_file_location = './{}/{}'.format(file_dir, file_name)
    absolute_file_location = os.path.join(os.path.dirname(__file__), relative_file_location)
    return absolute_file_location

# FROM STACK OVERFLOW
def dot(K, L):
    if len(K) != len(L):
        raise Exception("len(K) != leN(L)")
    return sum(i[0] * i[1] for i in zip(K, L))


def make_plot(file_dir, file_name, points):
    # print(points[0: 10])
    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]
    plt.plot(x_values, y_values)
    plt.savefig(get_file_path(file_dir, file_name))
    plt.clf()


def write_to_files(file_dir, file_name, lines):
    with open(get_file_path(file_dir, file_name), 'w+') as f:
        for line in lines:
            f.write('{}\n'.format(line))


def save_table_dict(file_dir, file_name, table_dict, m_or_n='n'):
    table_lines = list()
    table_header = list()
    table_header.append('{}/algorithm/estimated_margin/entry'.format(m_or_n))
    table_header.append('note: estimated_margin = 1 for winnow => estimated_margin > 0')
    table_header.append('------------------')
    for k, v in table_dict.items():
        table_lines.append('{}: {}'.format(k, v))
    table_lines.sort()
    write_to_files(file_dir, file_name, table_header + table_lines)

