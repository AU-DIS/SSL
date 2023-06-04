import math
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean

graph = 'football'
per = '0.1'
threshold = '0.2'
n_threshold = '0.2'
edge_removal = 3
measurement = 'balanced_accuracy'
prefix = ''

if len(sys.argv) >= 2:
    graph = sys.argv[1]
if len(sys.argv) >= 3:
    per = sys.argv[2]
if len(sys.argv) >= 4:
    threshold = sys.argv[3]
if len(sys.argv) >= 5:
    n_threshold = sys.argv[4]
if len(sys.argv) >= 6:
    edge_removal = int(sys.argv[5])
if len(sys.argv) >= 7:
    measurement = sys.argv[6]

if measurement == 'spectrum_diff':
    prefix = 'cc_'

def get_og_balanced_accuracy_file_string(folder):
    for _, _, files in os.walk(folder):
        for file in files:
            if 'og2' in file:
                # print("ENDED WITH OG2")
                return f"{folder}/og2_{measurement}.txt"
    # print("ENDED NORMALLY WITHOUT OG2")
    return f"{folder}/og_{measurement}.txt"

def get_data_from_folder(suffix, edge_removal):
    rootdir = f'experiments_final{suffix}/{graph}/{per}'
    directories = []

    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            directories.append(d)

    #Sort directories by name mapped to integer
    directories.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    conductance_list = []
    og_balanced_accuracy_list = []
    v_balanced_accuracy_list = []
    n_balanced_accuracy_list = []
    increase_v_balanced_accuracy_list = []
    increase_n_balanced_accuracy_list = []
    lowest_spectrum_balanced_accuracy_list = []

    # TODO husk at fix og balanced
    for folder in directories:
        with open(f'{folder}/conductance.txt') as f1, \
             open(get_og_balanced_accuracy_file_string(folder)) as f2, \
             open(f'{folder}/{measurement}_{threshold}.txt') as f3, \
             open(f'{folder}/n_{measurement}_{n_threshold}.txt') as f4, \
             open(f'{folder}/increasing_edge_removal/{measurement}_{threshold}.txt') as f5, \
             open(f'{folder}/increasing_edge_removal/n_{measurement}_{n_threshold}.txt') as f6:
            conductance = f1.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            og_balanced_accuracy = f2.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            v_balanced_accuracy = f3.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            n_balanced_accuracy = f4.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            increase_v_balanced_accuracy = f5.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            increase_n_balanced_accuracy = f6.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
        
        conductance = [float(i) for i in conductance]
        og_balanced_accuracy = [float(i) for i in og_balanced_accuracy]
        v_balanced_accuracy = [float(i) for i in v_balanced_accuracy]
        n_balanced_accuracy = [float(i) for i in n_balanced_accuracy]
        increase_v_balanced_accuracy = [float(i) for i in increase_v_balanced_accuracy]
        increase_n_balanced_accuracy = [float(i) for i in increase_n_balanced_accuracy]

        conductance_list.append(conductance[0])
        og_balanced_accuracy_list.append(og_balanced_accuracy[0])
        increase_v_balanced_accuracy_list.append(increase_v_balanced_accuracy[0])
        increase_n_balanced_accuracy_list.append(increase_n_balanced_accuracy[0])

        idx = edge_removal - 1 if len(v_balanced_accuracy) > 5 else edge_removal // 2
        v_balanced_accuracy_list.append(v_balanced_accuracy[idx])
        n_balanced_accuracy_list.append(n_balanced_accuracy[idx])

        # Find lowest spectrum diff for voting and neighborhood
        lowest_spectrum_diff = math.inf
        lowest_spectrum_treshold = ''
        lowest_spectrum_algorithm = ''
        lowest_spectrum_index = 0

        for i in range (2, 6):
            with open(f'{folder}/cc_spectrum_diff_0.{i}.txt') as f5, \
                 open(f'{folder}/cc_n_spectrum_diff_0.{i}.txt') as f6, \
                 open(f'{folder}/increasing_edge_removal/cc_{measurement}_0.{i}.txt') as f7, \
                 open(f'{folder}/increasing_edge_removal/cc_n_{measurement}_0.{i}.txt') as f8:
                v_spectrum_diff = f5.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
                n_spectrum_diff = f6.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
                increase_v_spectrum_diff = f7.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
                increase_n_spectrum_diff = f8.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            
            v_spectrum_diff = [float(i) for i in v_spectrum_diff]
            n_spectrum_diff = [float(i) for i in n_spectrum_diff]
            increase_v_spectrum_diff = [float(i) for i in increase_v_spectrum_diff]
            increase_n_spectrum_diff = [float(i) for i in increase_n_spectrum_diff]

            for j in range(len(v_spectrum_diff)):
                if v_spectrum_diff[j] < lowest_spectrum_diff:
                    # print(f"New lowest spectrum diff: {v_spectrum_diff[j]}, at index {j} for voting")
                    lowest_spectrum_diff = v_spectrum_diff[j]
                    lowest_spectrum_treshold = f'0.{i}'
                    lowest_spectrum_index = j
                    lowest_spectrum_algorithm = ''
                if n_spectrum_diff[j] < lowest_spectrum_diff:
                    # print(f"New lowest spectrum diff: {n_spectrum_diff[j]}, at index {j} for neighborhood")
                    lowest_spectrum_diff = n_spectrum_diff[j]
                    lowest_spectrum_treshold = f'0.{i}'
                    lowest_spectrum_index = j
                    lowest_spectrum_algorithm = 'n_'

        with open(f'{folder}/cc_{lowest_spectrum_algorithm}{measurement}_{lowest_spectrum_treshold}.txt') as f7:
            lowest_spectrum_balanced_accuracy = f7.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')

        lowest_spectrum_balanced_accuracy = [float(i) for i in lowest_spectrum_balanced_accuracy]

        lowest_spectrum_balanced_accuracy_list.append(lowest_spectrum_balanced_accuracy[lowest_spectrum_index])

    return conductance_list, og_balanced_accuracy_list, v_balanced_accuracy_list, n_balanced_accuracy_list, lowest_spectrum_balanced_accuracy_list, increase_v_balanced_accuracy_list, increase_n_balanced_accuracy_list

def plot(plt, conductance_list, balanced_accuracy_list, label, should_scatter = False):

    before = len(conductance_list)
    
    combined = zip(conductance_list, balanced_accuracy_list)

    if measurement == 'spectrum_diff':
        #if combined[1] == 999999 then remove this entry
        combined = [x for x in combined if x[1] != 999999]

    # print("This is combined initially", combined)
    sorted_combined = sorted(combined, key=lambda x: x[0])
    # print("This is combined afterwards", sorted_combined)
    x, y = zip(*sorted_combined)

    if measurement == 'spectrum_diff':
        after = len(x)
        print(f"Removed {before - after} out of {before} entries")
    

    x = np.array(x)
    y = np.array(y)

    """ x = np.array(conductance_list) """
    """ y = np.array(balanced_accuracy_list) """

    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    print(p)
    # if should_scatter:
    plt.scatter(x, y)
    plt.plot(x, p(x), label=label)

def entry_averaging():
    con1, og1, v1, n1, ls1, inc_v1, inc_n1 = get_data_from_folder("", edge_removal) 
    con2, og2, v2, n2, ls2, inc_v2, inc_n2 = get_data_from_folder("_2", edge_removal)
    con3, og3, v3, n3, ls3, inc_v3, inc_n3 = get_data_from_folder("_3", edge_removal) 
    con4, og4, v4, n4, ls4, inc_v4, inc_n4 = get_data_from_folder("_4", edge_removal) 
    con5, og5, v5, n5, ls5, inc_v5, inc_n5 = get_data_from_folder("_5", edge_removal) 

    _conductance_list = list(zip(con1, con2, con3, con4, con5))
    conductance_list = [mean(t) for t in _conductance_list]
    # print(_conductance_list)

    _og = list(zip(og1, og2, og3, og4, og5))
    og_balanced_accuracy_list = [mean(t) for t in _og]

    _v = list(zip(v1, v2, v3, v4, v5))
    v_balanced_accuracy_list = [mean(t) for t in _v]

    _n = list(zip(n1, n2, n3, n4, n5))
    n_balanced_accuracy_list = [mean(t) for t in _n]

    _inc_v = list(zip(inc_v1, inc_v2, inc_v3, inc_v4, inc_v5))
    increase_v_balanced_accuracy_list = [mean(t) for t in _inc_v]

    _inc_n = list(zip(inc_n1, inc_n2, inc_n3, inc_n4, inc_n5))
    increase_n_balanced_accuracy_list = [mean(t) for t in _inc_n]

    _ls = list(zip(ls1, ls2, ls3, ls4, ls5))
    lowest_spectrum_balanced_accuracy_list = [mean(t) for t in _ls]

    plt.plot(conductance_list, og_balanced_accuracy_list, marker='o', label='Original')
    plt.plot(conductance_list, v_balanced_accuracy_list, marker='o',label=f'Voting {edge_removal*10}% edges removed and {threshold} threshold')
    plt.plot(conductance_list, n_balanced_accuracy_list, marker='o',label=f'Neighborhood {edge_removal*10}% edges removed and {n_threshold} threshold')
    plt.plot(conductance_list, lowest_spectrum_balanced_accuracy_list, marker='o',label='Lowest Spectrum')
    plt.plot(conductance_list, increase_v_balanced_accuracy_list, marker='o', label=f'Voting increasing edge removal and threshold {threshold}')
    plt.plot(conductance_list, increase_n_balanced_accuracy_list, marker='o', label=f'Neighborhood increasing edge removal and threshold {n_threshold}')

def interval_averaging():
    data1 = list(zip(*get_data_from_folder("", edge_removal)))
    data2 = list(zip(*get_data_from_folder("_2", edge_removal)))
    data3 = list(zip(*get_data_from_folder("_3", edge_removal)))
    data4 = list(zip(*get_data_from_folder("_4", edge_removal)))
    data5 = list(zip(*get_data_from_folder("_5", edge_removal)))

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

def get_data_from_all_folders(edge_removal):
    conductance_list = []
    og_balanced_accuracy_list = []
    v_balanced_accuracy_list = []
    n_balanced_accuracy_list = []
    lowest_spectrum_balanced_accuracy_list = []
    increase_v_balanced_accuracy_list = []
    increase_n_balanced_accuracy_list = []
    lowest_spectrum_balanced_accuracy_list = []

    con_list, og_list, v_list, n_list, ls_list, inc_v, inc_n = get_data_from_folder("", edge_removal)
    conductance_list += con_list
    og_balanced_accuracy_list += og_list
    v_balanced_accuracy_list += v_list
    n_balanced_accuracy_list += n_list
    increase_v_balanced_accuracy_list += inc_v
    increase_n_balanced_accuracy_list += inc_n
    lowest_spectrum_balanced_accuracy_list += ls_list

    con_list, og_list, v_list, n_list, ls_list, inc_v, inc_n  = get_data_from_folder("_2", edge_removal)
    conductance_list += con_list
    og_balanced_accuracy_list += og_list
    v_balanced_accuracy_list += v_list
    n_balanced_accuracy_list += n_list
    increase_v_balanced_accuracy_list += inc_v
    increase_n_balanced_accuracy_list += inc_n
    lowest_spectrum_balanced_accuracy_list += ls_list

    con_list, og_list, v_list, n_list, ls_list, inc_v, inc_n  = get_data_from_folder("_3", edge_removal)
    conductance_list += con_list
    og_balanced_accuracy_list += og_list
    v_balanced_accuracy_list += v_list
    n_balanced_accuracy_list += n_list
    increase_v_balanced_accuracy_list += inc_v
    increase_n_balanced_accuracy_list += inc_n
    lowest_spectrum_balanced_accuracy_list += ls_list

    con_list, og_list, v_list, n_list, ls_list, inc_v, inc_n  = get_data_from_folder("_4", edge_removal)
    conductance_list += con_list
    og_balanced_accuracy_list += og_list
    v_balanced_accuracy_list += v_list
    n_balanced_accuracy_list += n_list
    increase_v_balanced_accuracy_list += inc_v
    increase_n_balanced_accuracy_list += inc_n
    lowest_spectrum_balanced_accuracy_list += ls_list

    con_list, og_list, v_list, n_list, ls_list, inc_v, inc_n  = get_data_from_folder("_5", edge_removal)
    conductance_list += con_list
    og_balanced_accuracy_list += og_list
    v_balanced_accuracy_list += v_list
    n_balanced_accuracy_list += n_list
    increase_v_balanced_accuracy_list += inc_v
    increase_n_balanced_accuracy_list += inc_n
    lowest_spectrum_balanced_accuracy_list += ls_list

    return conductance_list, og_balanced_accuracy_list, v_balanced_accuracy_list, n_balanced_accuracy_list, lowest_spectrum_balanced_accuracy_list, increase_v_balanced_accuracy_list, increase_n_balanced_accuracy_list


def regression():
    conductance_list, og_balanced_accuracy_list, v_balanced_accuracy_list, n_balanced_accuracy_list, lowest_spectrum_balanced_accuracy_list, increase_v_balanced_accuracy_list, increase_n_balanced_accuracy_list = get_data_from_all_folders(edge_removal)

    if measurement == 'balanced_accuracy':
        ones = [1, 1, 1, 1, 1]
        conductance_list = [0, 0, 0, 0, 0] + conductance_list
        og_balanced_accuracy_list = ones + og_balanced_accuracy_list
        v_balanced_accuracy_list = ones + v_balanced_accuracy_list
        n_balanced_accuracy_list = ones + n_balanced_accuracy_list
        increase_v_balanced_accuracy_list = ones + increase_v_balanced_accuracy_list
        increase_n_balanced_accuracy_list = ones + increase_n_balanced_accuracy_list
        lowest_spectrum_balanced_accuracy_list = ones + lowest_spectrum_balanced_accuracy_list

    elif measurement == 'spectrum_diff':
        zeros = [0, 0, 0, 0, 0]
        conductance_list = zeros + conductance_list
        og_balanced_accuracy_list = zeros + og_balanced_accuracy_list
        v_balanced_accuracy_list = zeros + v_balanced_accuracy_list
        n_balanced_accuracy_list = zeros + n_balanced_accuracy_list
        increase_v_balanced_accuracy_list = zeros + increase_v_balanced_accuracy_list
        increase_n_balanced_accuracy_list = zeros + increase_n_balanced_accuracy_list
        lowest_spectrum_balanced_accuracy_list = zeros + lowest_spectrum_balanced_accuracy_list

    plot(plt, conductance_list, og_balanced_accuracy_list, "Original")
    plot(plt, conductance_list, v_balanced_accuracy_list, f'Voting {edge_removal*10}% edges removed and {threshold} threshold')
    plot(plt, conductance_list, n_balanced_accuracy_list, f'Neighborhood {edge_removal*10}% edges removed and {n_threshold} threshold')
    plot(plt, conductance_list, lowest_spectrum_balanced_accuracy_list, "Lowest Spectrum")
    plot(plt, conductance_list, increase_v_balanced_accuracy_list, f'Voting increasing edge removal and threshold {threshold}')
    plot(plt, conductance_list, increase_n_balanced_accuracy_list, f'Neighborhood increasing edge removal and threshold {n_threshold}')

    plt.xlabel('Conductance')
    plt.ylabel(f'{measurement}')
    plt.title(f'{measurement} vs Conductance for {graph} with |V|/|V_Q|={per}')

def plot_edge_removals():
    _, _, v_balanced_accuracy_list_10, n_balanced_accuracy_list_10, _, _, _ = get_data_from_all_folders(edge_removal=1)
    _, _, v_balanced_accuracy_list_30, n_balanced_accuracy_list_30, _, _, _ = get_data_from_all_folders(edge_removal=3)
    _, _, v_balanced_accuracy_list_50, n_balanced_accuracy_list_50, _, _, _ = get_data_from_all_folders(edge_removal=5)
    _, _, v_balanced_accuracy_list_70, n_balanced_accuracy_list_70, _, _, _ = get_data_from_all_folders(edge_removal=7)
    _, _, v_balanced_accuracy_list_90, n_balanced_accuracy_list_90, _, _, _ = get_data_from_all_folders(edge_removal=9)

    print(v_balanced_accuracy_list_10)

    x = [10, 30, 50, 70, 90]
    y1 = [np.mean(v_balanced_accuracy_list_10), np.mean(v_balanced_accuracy_list_30), np.mean(v_balanced_accuracy_list_50), np.mean(v_balanced_accuracy_list_70), np.mean(v_balanced_accuracy_list_90)]
    y2 = [np.mean(n_balanced_accuracy_list_10), np.mean(n_balanced_accuracy_list_30), np.mean(n_balanced_accuracy_list_50), np.mean(n_balanced_accuracy_list_70), np.mean(n_balanced_accuracy_list_90)]
    plt.plot(x, y1, marker='o',label=f'Voting {edge_removal*10}% edges removed and {threshold} threshold')
    plt.plot(x, y2, marker='o',label=f'Neighborhood {edge_removal*10}% edges removed and {n_threshold} threshold')

    plt.xlabel('Edge Removal %')
    plt.ylabel('Balanced Accuracy')
    plt.title(f'Balanced Accuracy vs edge removal % for {graph} with |V|/|V_Q|={per}')

if __name__ == '__main__':
    # entry_averaging()
    # interval_averaging()
    regression()
    # plot_edge_removals()

    
    plt.legend()
    plt.show()
