import math
import sys
import os
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    rootdir = f'experiments_final/{graph}/{per}'
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

    for folder in directories:
        with open(f'{folder}/conductance.txt') as f1, \
             open(f'{folder}/og_balanced_accuracy.txt') as f2, \
             open(f'{folder}/balanced_accuracy_{threshold}.txt') as f3, \
             open(f'{folder}/n_balanced_accuracy_{threshold}.txt') as f4, \
             open(f'{folder}/increasing_edge_removal/balanced_accuracy_{threshold}.txt') as f5, \
             open(f'{folder}/increasing_edge_removal/n_balanced_accuracy_{threshold}.txt') as f6:
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
        v_balanced_accuracy_list.append(v_balanced_accuracy[edge_removal-1])
        n_balanced_accuracy_list.append(n_balanced_accuracy[edge_removal-1])
        increase_v_balanced_accuracy_list.append(increase_v_balanced_accuracy[0])
        increase_n_balanced_accuracy_list.append(increase_n_balanced_accuracy[0])

        # Find lowest spectrum diff for voting and neighborhood
        lowest_spectrum_diff = math.inf
        lowest_spectrum_treshold = ''
        lowest_spectrum_algorithm = ''
        lowest_spectrum_index = 0

        for i in range (2, 4):
            with open(f'{folder}/cc_spectrum_diff_0.{i}.txt') as f5, \
                 open(f'{folder}/cc_n_spectrum_diff_0.{i}.txt') as f6, \
                 open(f'{folder}/increasing_edge_removal/cc_balanced_accuracy_0.{i}.txt') as f7, \
                 open(f'{folder}/increasing_edge_removal/cc_n_balanced_accuracy_0.{i}.txt') as f8:
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
                    lowest_spectrum_diff = v_spectrum_diff[j]
                    lowest_spectrum_treshold = f'0.{i}'
                    lowest_spectrum_index = j
                    lowest_spectrum_algorithm = 'cc'
                if n_spectrum_diff[j] < lowest_spectrum_diff:
                    lowest_spectrum_diff = n_spectrum_diff[j]
                    lowest_spectrum_treshold = f'0.{i}'
                    lowest_spectrum_index = j
                    lowest_spectrum_algorithm = 'cc_n'
            if increase_v_spectrum_diff[0] < lowest_spectrum_diff:
                lowest_spectrum_diff = increase_v_spectrum_diff[0]
                lowest_spectrum_treshold = f'0.{i}'
                lowest_spectrum_index = 0
                lowest_spectrum_algorithm = 'increasing_edge_removal/cc'
            if increase_n_spectrum_diff[0] < lowest_spectrum_diff:
                lowest_spectrum_diff = increase_n_spectrum_diff[0]
                lowest_spectrum_treshold = f'0.{i}'
                lowest_spectrum_index = 0
                lowest_spectrum_algorithm = 'increasing_edge_removal/cc_n'

        with open(f'{folder}/{lowest_spectrum_algorithm}_balanced_accuracy_{lowest_spectrum_treshold}.txt') as f7:
            lowest_spectrum_balanced_accuracy = f7.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')

        lowest_spectrum_balanced_accuracy = [float(i) for i in lowest_spectrum_balanced_accuracy]

        lowest_spectrum_balanced_accuracy_list.append(lowest_spectrum_balanced_accuracy[lowest_spectrum_index])

    plt.plot(conductance_list, og_balanced_accuracy_list, marker='o', label='Original')
    plt.plot(conductance_list, v_balanced_accuracy_list, marker='o',label=f'Voting {edge_removal*10}% edge removal and threshold {threshold}')
    plt.plot(conductance_list, n_balanced_accuracy_list, marker='o',label=f'Neighborhood {edge_removal*10}% edge removal and threshold {threshold}')
    plt.plot(conductance_list, increase_v_balanced_accuracy_list, marker='o',label=f'Voting increasing edge removal and threshold {threshold}')
    plt.plot(conductance_list, increase_n_balanced_accuracy_list, marker='o',label=f'Neighborhood increasing edge removal and threshold {threshold}')
    plt.plot(conductance_list, lowest_spectrum_balanced_accuracy_list, marker='o',label='Lowest Spectrum')
    plt.xlabel('Conductance')
    plt.ylabel('Balanced Accuracy')
    plt.title(f'Balanced Accuracy vs Conductance for {graph} with |V|/|V_Q|={per}')
    plt.legend()
    plt.show()