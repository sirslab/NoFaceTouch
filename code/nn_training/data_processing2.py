import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import sys
from params import *

data_path = "raw_data_processing/"

labelling_file = sys.argv[1]
sensor_file = sys.argv[2]
split = sys.argv[3]

# python data_processing2.py train0_labelling.txt train0.csv train
# python data_processing2.py test0_labelling.txt test0.csv test


def data_selection():
    reading = pd.read_csv(data_path + labelling_file, delimiter=':')
    readings = reading.to_numpy()

    offset = readings[np.where(readings == "OFFSET")[0], [1]]
    readings[:, 1] = readings[:, 1] - offset
    contacts_start = list(readings[np.where(readings == "CONTACT_START")[0], [1]])
    contacts_stop = list(readings[np.where(readings == "CONTACT_STOP")[0], [1]])
    neutrals = list(readings[np.where(readings == "NEUTRAL")[0], [1]])

    return contacts_start, contacts_stop, neutrals


def belong_to(elem, intervals):
    belongs = False

    for interval in intervals:
        if interval[0] < elem < interval[2]:
            belongs = True
            break

    return belongs, interval


def files_generation(raw_data):
    print('processing ', raw_data)
    read_file = pd.read_csv(raw_data)

    data_all = read_file.to_numpy()
    print(data_all.shape)

    data_all[:, 0] -= data_all[0, 0]
    data_all[:, 0] /= 1e9
    print(data_all[:, 0])

    contacts_start, contacts_stop, neutrals = data_selection()

    temp = [0, 0]

    intervals = []
    for i in range(0, len(neutrals)):
        intervals.append((neutrals[i], contacts_start[i], contacts_stop[i]))

    i = 0
    for t in data_all[:, 0]:

        belongs, interval = belong_to(t, intervals)

        if belongs and interval != temp:

            d = data_all[np.logical_and(data_all[:, 0] > interval[0], data_all[:, 0] < interval[1])][:, 0: ]
            np.savetxt(dataset_path + split + "/" + sensor_file.replace(".csv", "_") + str(i) + "_pre.txt", d)

            d = data_all[np.logical_and(data_all[:, 0] >= interval[1], data_all[:, 0] < interval[2])][:, 0: ]
            np.savetxt(dataset_path + split + "/" + sensor_file.replace(".csv", "_") + str(i) + "_post.txt", d)

            temp = interval

            i = i + 1


files_generation(data_path + sensor_file)
