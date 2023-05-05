import math
import sys
import os
import matplotlib.pyplot as plt

graph = 'football'
per = '0.1'
threshold = '0.3'
edge_removal = 3

if len(sys.argv) >= 2:
    graph = sys.argv[1]
if len(sys.argv) >= 3:
    per = sys.argv[2]
if len(sys.argv) >= 4:
    threshold = sys.argv[3]
if len(sys.argv) >= 5:
    edge_removal = int(sys.argv[4][2])

if __name__ == '__main__':
    rootdir = f'experiments_edge_removal/{graph}/{per}'
    directories = []

    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            directories.append(d)

    directories.sort()

    conductance_list = []
    og_balanced_accuracy_list = []
    v_balanced_accuracy_list = []
    n_balanced_accuracy_list = []
    v_lowest_spectrum_balanced_accuracy_list = []
    n_lowest_spectrum_balanced_accuracy_list = []

    for folder in directories:
        with open(f'{folder}/conductance.txt') as f1, \
             open(f'{folder}/og_balanced_accuracy.txt') as f2, \
             open(f'{folder}/balanced_accuracy_{threshold}.txt') as f3, \
             open(f'{folder}/n_balanced_accuracy_{threshold}.txt') as f4:  
            conductance = f1.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            og_balanced_accuracy = f2.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            v_balanced_accuracy = f3.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            n_balanced_accuracy = f4.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
        
        v_balanced_accuracy = [float(i) for i in v_balanced_accuracy]
        n_balanced_accuracy = [float(i) for i in n_balanced_accuracy]
        og_balanced_accuracy = [float(i) for i in og_balanced_accuracy]
        conductance = [float(i) for i in conductance]

        conductance_list.append(conductance[0])
        og_balanced_accuracy_list.append(og_balanced_accuracy[0])
        v_balanced_accuracy_list.append(v_balanced_accuracy[edge_removal-1])
        n_balanced_accuracy_list.append(n_balanced_accuracy[edge_removal-1])

        # Find lowest spectrum diff for voting and neighborhood
        n_lowest_spectrum_diff = math.inf
        n_lowest_spectrum_treshold = ''
        n_lowest_spectrum_index = 0

        v_lowest_spectrum_diff = math.inf
        v_lowest_spectrum_treshold = ''
        v_lowest_spectrum_index = 0

        for i in range (1, 7):
            with open(f'{folder}/spectrum_diff_0.{i}.txt') as f5, \
                 open(f'{folder}/n_spectrum_diff_0.{i}.txt') as f6:
                v_spectrum_diff = f5.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
                n_spectrum_diff = f6.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            
            v_spectrum_diff = [float(i) for i in v_spectrum_diff]
            n_spectrum_diff = [float(i) for i in n_spectrum_diff]

            for j in range(len(v_spectrum_diff)):
                if v_spectrum_diff[j] < v_lowest_spectrum_diff:
                    v_lowest_spectrum_diff = v_spectrum_diff[j]
                    v_lowest_spectrum_treshold = f'0.{i}'
                    v_lowest_spectrum_index = j
                if n_spectrum_diff[j] < n_lowest_spectrum_diff:
                    n_lowest_spectrum_diff = n_spectrum_diff[j]
                    n_lowest_spectrum_treshold = f'0.{i}'
                    n_lowest_spectrum_index = j

        with open(f'{folder}/balanced_accuracy_{v_lowest_spectrum_treshold}.txt') as f7, \
             open(f'{folder}/n_balanced_accuracy_{n_lowest_spectrum_treshold}.txt') as f8:
            v_lowest_spectrum_balanced_accuracy = f7.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            n_lowest_spectrum_balanced_accuracy = f8.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')

        v_lowest_spectrum_balanced_accuracy = [float(i) for i in v_lowest_spectrum_balanced_accuracy]
        n_lowest_spectrum_balanced_accuracy = [float(i) for i in n_lowest_spectrum_balanced_accuracy]

        v_lowest_spectrum_balanced_accuracy_list.append(v_lowest_spectrum_balanced_accuracy[v_lowest_spectrum_index])
        n_lowest_spectrum_balanced_accuracy_list.append(n_lowest_spectrum_balanced_accuracy[n_lowest_spectrum_index])

    plt.plot(conductance_list, og_balanced_accuracy_list, marker='o', label='Original')
    plt.plot(conductance_list, v_balanced_accuracy_list, marker='o',label='Voting')
    plt.plot(conductance_list, n_balanced_accuracy_list, marker='o',label='Neighborhood')
    plt.plot(conductance_list, v_lowest_spectrum_balanced_accuracy_list, marker='o',label='Voting Lowest Spectrum')
    plt.plot(conductance_list, n_lowest_spectrum_balanced_accuracy_list, marker='o',label='Neighborhood Lowest Spectrum')
    plt.xlabel('Conductance')
    plt.ylabel('Balanced Accuracy')
    plt.title(f'Balanced Accuracy vs Conductance for {graph} with |V|/|V_Q|={per}')
    plt.legend()
    plt.show()
