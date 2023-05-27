import math
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
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
    print("This is combined initially", combined)
    sorted_combined = sorted(combined, key=lambda x: x[0])
    print("This is combined afterwards", sorted_combined)
    x, y = zip(*sorted_combined)
    
    x = np.array(x)
    y = np.array(y)

    """ x = np.array(conductance_list) """
    """ y = np.array(balanced_accuracy_list) """

    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
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
    plt.plot(conductance_list, v_f1_list, marker='o',label=f'Voting {edge_removal*10}% edges removed and {threshold} threshold')
    # plt.plot(conductance_list, n_f1_list, marker='o',label=f'Neighborhood {edge_removal*10}% edges removed and {threshold} threshold')

def interval_averaging():
    data1 = list(zip(*get_data_from_folder("")))
    data2 = list(zip(*get_data_from_folder("_2")))
    data3 = list(zip(*get_data_from_folder("_3")))
    data4 = list(zip(*get_data_from_folder("_4")))
    data5 = list(zip(*get_data_from_folder("_5")))

    all_data = sorted(data1 + data2 + data3 + data4 + data5) # sorted by conductance!

    folders = 5
    subfolders = 10

    conductance_list = []
    og_balanced_accuracy_list = []
    v_balanced_accuracy_list = []
    n_balanced_accuracy_list = []
    lowest_spectrum_balanced_accuracy_list = []
    increase_v_balanced_accuracy_list = []
    increase_n_balanced_accuracy_list = []

    for i in range(subfolders):
        lower = i * folders
        upper = (i+1) * folders
        part_data = all_data[lower:upper]
        con_mean, og_mean, v_mean, n_mean, ls_mean, inc_v_mean, inc_n_mean = [sum(column) / folders for column in zip(*part_data)]
        conductance_list.append(con_mean)
        og_balanced_accuracy_list.append(og_mean)
        v_balanced_accuracy_list.append(v_mean)
        n_balanced_accuracy_list.append(n_mean)
        lowest_spectrum_balanced_accuracy_list.append(ls_mean)
        increase_v_balanced_accuracy_list.append(inc_v_mean)
        increase_n_balanced_accuracy_list.append(inc_n_mean)

    plt.plot(conductance_list, og_balanced_accuracy_list, marker='o', label='Original')
    plt.plot(conductance_list, v_balanced_accuracy_list, marker='o',label=f'Voting {edge_removal*10}% edges removed and {threshold} threshold')
    plt.plot(conductance_list, n_balanced_accuracy_list, marker='o',label=f'Neighborhood {edge_removal*10}% edges removed and {threshold} threshold')
    plt.plot(conductance_list, lowest_spectrum_balanced_accuracy_list, marker='o',label='Lowest Spectrum')
    plt.plot(conductance_list, increase_v_balanced_accuracy_list, marker='o', label=f'Voting increasing edge removal and threshold {threshold}')
    plt.plot(conductance_list, increase_n_balanced_accuracy_list, marker='o', label=f'Neighborhood increasing edge removal and threshold {threshold}')

def regression():
    conductance_list = []
    og_balanced_accuracy_list = []
    v_balanced_accuracy_list = []
    n_balanced_accuracy_list = []
    lowest_spectrum_balanced_accuracy_list = []
    increase_v_balanced_accuracy_list = []
    increase_n_balanced_accuracy_list = []
    lowest_spectrum_balanced_accuracy_list = []

    con_list, og_list, v_list, n_list, ls_list, inc_v, inc_n = get_data_from_folder("")
    conductance_list += con_list
    og_balanced_accuracy_list += og_list
    v_balanced_accuracy_list += v_list
    n_balanced_accuracy_list += n_list
    increase_v_balanced_accuracy_list += inc_v
    increase_n_balanced_accuracy_list += inc_n
    lowest_spectrum_balanced_accuracy_list += ls_list

    con_list, og_list, v_list, n_list, ls_list, inc_v, inc_n  = get_data_from_folder("_2")
    conductance_list += con_list
    og_balanced_accuracy_list += og_list
    v_balanced_accuracy_list += v_list
    n_balanced_accuracy_list += n_list
    increase_v_balanced_accuracy_list += inc_v
    increase_n_balanced_accuracy_list += inc_n
    lowest_spectrum_balanced_accuracy_list += ls_list

    con_list, og_list, v_list, n_list, ls_list, inc_v, inc_n  = get_data_from_folder("_3")
    conductance_list += con_list
    og_balanced_accuracy_list += og_list
    v_balanced_accuracy_list += v_list
    n_balanced_accuracy_list += n_list
    increase_v_balanced_accuracy_list += inc_v
    increase_n_balanced_accuracy_list += inc_n
    lowest_spectrum_balanced_accuracy_list += ls_list

    con_list, og_list, v_list, n_list, ls_list, inc_v, inc_n  = get_data_from_folder("_4")
    conductance_list += con_list
    og_balanced_accuracy_list += og_list
    v_balanced_accuracy_list += v_list
    n_balanced_accuracy_list += n_list
    increase_v_balanced_accuracy_list += inc_v
    increase_n_balanced_accuracy_list += inc_n
    lowest_spectrum_balanced_accuracy_list += ls_list

    con_list, og_list, v_list, n_list, ls_list, inc_v, inc_n  = get_data_from_folder("_5")
    conductance_list += con_list
    og_balanced_accuracy_list += og_list
    v_balanced_accuracy_list += v_list
    n_balanced_accuracy_list += n_list
    increase_v_balanced_accuracy_list += inc_v
    increase_n_balanced_accuracy_list += inc_n
    lowest_spectrum_balanced_accuracy_list += ls_list

    plot(plt, conductance_list, og_balanced_accuracy_list, "Original")
    plot(plt, conductance_list, v_balanced_accuracy_list, f'Voting {edge_removal*10}% edges removed and {threshold} threshold')
    plot(plt, conductance_list, n_balanced_accuracy_list, f'Neighborhood {edge_removal*10}% edges removed and {threshold} threshold')
    plot(plt, conductance_list, lowest_spectrum_balanced_accuracy_list, "Lowest Spectrum")
    plot(plt, conductance_list, increase_v_balanced_accuracy_list, f'Voting increasing edge removal and threshold {threshold}')
    plot(plt, conductance_list, increase_n_balanced_accuracy_list, f'Neighborhood increasing edge removal and threshold {threshold}')

def plot_multiple_thresholds():
    entry_averaging('0.1')
    entry_averaging('0.2')
    entry_averaging('0.3')
    entry_averaging('0.4')
    entry_averaging('0.5')
    entry_averaging('0.6')

if __name__ == '__main__':
    plot_multiple_thresholds()
    # entry_averaging()
    # interval_averaging()
    # regression()

    plt.xlabel('Conductance')
    plt.ylabel('f1-score')
    plt.title(f'f1-score vs Conductance for {graph} with |V|/|V_Q|={per}')
    plt.legend()
    plt.show()
