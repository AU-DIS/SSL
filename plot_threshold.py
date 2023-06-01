import math
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
from statistics import mean

graph = 'football'
per = '0.1'
threshold = '0.2'
edge_removal = 3

if len(sys.argv) >= 2:
    graph = sys.argv[1]
if len(sys.argv) >= 3:
    per = sys.argv[2]
if len(sys.argv) >= 4:
    threshold = sys.argv[3]
if len(sys.argv) >= 5:
    edge_removal = int(sys.argv[4])

def get_og_prefix(folder):
    for _, _, files in os.walk(folder):
        for file in files:
            if 'og2' in file:
                return "og2"
    return "og"

def get_data_from_folder(suffix, threshold):
    rootdir = f'experiments_final{suffix}/{graph}/{per}'
    directories = []

    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            directories.append(d)

    #Sort directories by name mapped to integer
    directories.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    conductance_list = []
    og_f1_list = []
    v_f1_list = []
    n_f1_list = []

    for folder in directories:
        with open(f'{folder}/conductance.txt') as f1, \
             open(f'{folder}/{get_og_prefix(folder)}_f1.txt') as f2, \
             open(f'{folder}/f1_{threshold}.txt') as f3, \
             open(f'{folder}/n_f1_{threshold}.txt') as f4:
            conductance = f1.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            og_f1 = f2.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            v_f1 = f3.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            n_f1 = f4.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
        
        conductance = [float(i) for i in conductance]
        og_f1 = [float(i) for i in og_f1]
        v_f1 = [float(i) for i in v_f1]
        n_f1 = [float(i) for i in n_f1]

        conductance_list.append(conductance[0])
        og_f1_list.append(og_f1[0])

        idx = edge_removal - 1 if len(v_f1) > 5 else edge_removal // 2
        v_f1_list.append(v_f1[idx])
        n_f1_list.append(n_f1[idx])

    return conductance_list, og_f1_list, v_f1_list, n_f1_list

def plot(plt, conductance_list, f1_list, label, should_scatter = False):

    combined = zip(conductance_list, f1_list)
    sorted_combined = sorted(combined, key=lambda x: x[0])
    x, y = zip(*sorted_combined)
    
    x = np.array(x)
    y = np.array(y)

    """ x = np.array(conductance_list) """
    """ y = np.array(balanced_accuracy_list) """

    z = np.polyfit(x, y, 2)
    # print(z)
    p = np.poly1d(z)
    print(p)
    if should_scatter:
        plt.scatter(x, y)
    plt.plot(x, p(x), label=label)

def entry_averaging(threshold):
    con1, og1, v1, n1 = get_data_from_folder("", threshold) 
    con2, og2, v2, n2 = get_data_from_folder("_2", threshold) 
    con3, og3, v3, n3 = get_data_from_folder("_3", threshold) 
    con4, og4, v4, n4 = get_data_from_folder("_4", threshold) 
    con5, og5, v5, n5 = get_data_from_folder("_5", threshold) 

    _conductance_list = list(zip(con1, con2, con3, con4, con5))
    conductance_list = [mean(t) for t in _conductance_list]
    # print(_conductance_list)

    _og = list(zip(og1, og2, og3, og4, og5))
    og_f1_list = [mean(t) for t in _og]

    _v = list(zip(v1, v2, v3, v4, v5))
    v_f1_list = [mean(t) for t in _v]

    _n = list(zip(n1, n2, n3, n4, n5))
    n_f1_list = [mean(t) for t in _n]

    # plt.plot(conductance_list, og_f1_list, marker='o', label='Original')
    # plt.plot(conductance_list, v_f1_list, marker='o',label=f'Voting {edge_removal*10}% edges removed and {threshold} threshold')
    # plt.plot(conductance_list, n_f1_list, marker='o',label=f'Neighborhood {edge_removal*10}% edges removed and {threshold} threshold')
    return conductance_list, og_f1_list, v_f1_list, n_f1_list

def regression(threshold):
    conductance_list = []
    og_f1_list = []
    v_f1_list = []
    n_f1_list = []

    con_list, og_list, v_list, n_list = get_data_from_folder("", threshold)
    conductance_list += con_list
    og_f1_list += og_list
    v_f1_list += v_list
    n_f1_list += n_list

    con_list, og_list, v_list, n_list = get_data_from_folder("_2", threshold)
    conductance_list += con_list
    og_f1_list += og_list
    v_f1_list += v_list
    n_f1_list += n_list

    con_list, og_list, v_list, n_list = get_data_from_folder("_3", threshold)
    conductance_list += con_list
    og_f1_list += og_list
    v_f1_list += v_list
    n_f1_list += n_list

    con_list, og_list, v_list, n_list = get_data_from_folder("_4", threshold)
    conductance_list += con_list
    og_f1_list += og_list
    v_f1_list += v_list
    n_f1_list += n_list

    con_list, og_list, v_list, n_list = get_data_from_folder("_5", threshold)
    conductance_list += con_list
    og_f1_list += og_list
    v_f1_list += v_list
    n_f1_list += n_list

    # conductance_list = [0, 0, 0, 0, 0] + conductance_list
    # og_f1_list = [1, 1, 1, 1, 1] + og_f1_list
    # v_f1_list = [1, 1, 1, 1, 1] + v_f1_list
    # n_f1_list = [1, 1, 1, 1, 1] + n_f1_list

    # print(v_f1_list)

    # plot(plt, conductance_list, og_f1_list, "Original")
    print("Threshold: ", threshold)
    plot(plt, conductance_list, v_f1_list, f'Voting {edge_removal*10}% edges removed and {threshold} threshold')
    # plot(plt, conductance_list, n_f1_list, f'Neighborhood {edge_removal*10}% edges removed and {threshold} threshold')

def plot_multiple_thresholds():
    regression('0.1')
    regression('0.2')
    regression('0.3')
    regression('0.4')
    regression('0.5')
    regression('0.6')

    plt.xlabel('Conductance')
    plt.ylabel('f1-score')
    plt.title(f'f1-score vs Conductance for {graph} with |V|/|V_Q|={per}')
    plt.legend()
    plt.show()

def plot_f1_vs_thresholds():
    con_1, og_f1_1, v_f1_1, n_f1_1 = entry_averaging(0.1)
    con_2, og_f1_2, v_f1_2, n_f1_2 = entry_averaging(0.2)
    con_3, og_f1_3, v_f1_3, n_f1_3 = entry_averaging(0.3)
    con_4, og_f1_4, v_f1_4, n_f1_4 = entry_averaging(0.4)
    con_5, og_f1_5, v_f1_5, n_f1_5 = entry_averaging(0.5)
    con_6, og_f1_6, v_f1_6, n_f1_6 = entry_averaging(0.6)

    for i in range (1, 8, 3):

        x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        y0 = [og_f1_1[i], og_f1_2[i], og_f1_3[i], og_f1_4[i], og_f1_5[i], og_f1_6[i]]
        y1 = [v_f1_1[i], v_f1_2[i], v_f1_3[i], v_f1_4[i], v_f1_5[i], v_f1_6[i]]
        y2 = [n_f1_1[i], n_f1_2[i], n_f1_3[i], n_f1_4[i], n_f1_5[i], n_f1_6[i]]

        plt.plot(x, y0, label='Original')
        # plot(plt, x, y0, 'Original', True)
        # plot(plt, x, y3, f'Voting {edge_removal*10}% edges removed', True)

        plt.plot(x, y1, marker='o', label=f'Voting {edge_removal*10}% edges removed')
        plt.plot(x, y2, marker='o', label=f'Neighborhood {edge_removal*10}% edges removed')

        plt.show()

        # with open(f'Threshold_vs_f1_con={round(con_1[i], 2)}_removal={edge_removal/10}_{graph}_{per}.csv', 'w', encoding='UTF8', newline='') as csv_file:
        #     writer = csv.writer(csv_file, delimiter=" ")
        #     writer.writerow(['threshold', 'edge_removal', 'conductance', 'ssl', 'voting', 'neighborhood'])
        #     for j in range(len(x)):
        #         writer.writerow([x[j], edge_removal//10, con_1[i], y0[j], y1[j], y2[j]])

if __name__ == '__main__':
    # plot_multiple_thresholds()
    plot_f1_vs_thresholds()


