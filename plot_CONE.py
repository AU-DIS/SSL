import math
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
from statistics import mean

graph = 'football'
per = '0.1'

if len(sys.argv) >= 2:
    graph = sys.argv[1]
if len(sys.argv) >= 3:
    per = sys.argv[2]

def get_data_from_folder(suffix):
    rootdir = f'CONE/experiments_CONE/{suffix}/{graph}/{per}'
    
    with open(f'{rootdir}/conductances.txt') as f1, \
         open(f'{rootdir}/balanced_accuracies.txt') as f2:
        conductance = f1.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
        balanced_accuracies = f2.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
     
        conductance = [float(i) for i in conductance]
        balanced_accuracies = [float(i) for i in balanced_accuracies]

    return conductance, balanced_accuracies

def plot(plt, conductance_list, balanced_accuracy_list, label, should_scatter = False):

    combined = zip(conductance_list, balanced_accuracy_list)
    sorted_combined = sorted(combined, key=lambda x: x[0])
    x, y = zip(*sorted_combined)

    x = np.array(x)
    y = np.array(y)

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    print(p)
    # if should_scatter:
    plt.scatter(x, y)
    plt.plot(x, p(x), label=label)

def get_data_from_all_folders():
    conductance_list = []
    balanced_accuracy_list = []

    con_list, bal_list = get_data_from_folder(1)
    conductance_list += con_list
    balanced_accuracy_list += bal_list

    con_list, bal_list = get_data_from_folder(2)
    conductance_list += con_list
    balanced_accuracy_list += bal_list

    con_list, bal_list = get_data_from_folder(3)
    conductance_list += con_list
    balanced_accuracy_list += bal_list

    con_list, bal_list = get_data_from_folder(4)
    conductance_list += con_list
    balanced_accuracy_list += bal_list

    con_list, bal_list = get_data_from_folder(5)
    conductance_list += con_list
    balanced_accuracy_list += bal_list

    return conductance_list, balanced_accuracy_list


def regression():
    conductance_list, balanced_accuracy_list = get_data_from_all_folders()

    plot(plt, conductance_list, balanced_accuracy_list, "CONE")
    # plot(plt, conductance_list, v_balanced_accuracy_list, f'Voting {edge_removal*10}% edges removed and {threshold} threshold')
    # plot(plt, conductance_list, n_balanced_accuracy_list, f'Neighborhood {edge_removal*10}% edges removed and {n_threshold} threshold')

    plt.xlabel('Conductance')
    plt.ylabel('Balanced Accuracy')
    plt.title(f'CONE - Balaned accuracy vs Conductance for {graph} with |V|/|V_Q|={per}')

if __name__ == '__main__':
    regression()

    plt.legend()
    plt.show()
