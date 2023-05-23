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
    og_spectrum_diff_list = []
    v_spectrum_diff_list = []
    n_spectrum_diff_list = []
    cv_spectrum_diff_list = []
    cn_spectrum_diff_list = []
    lowest_spectrum_spectrum_diff_list = []

    for folder in directories:
        with open(f'{folder}/conductance.txt') as f1, \
             open(f'{folder}/og_spectrum_diff.txt') as f2, \
             open(f'{folder}/spectrum_diff_{threshold}.txt') as f3, \
             open(f'{folder}/n_spectrum_diff_{threshold}.txt') as f4, \
             open(f'{folder}/cc_spectrum_diff_{threshold}.txt') as f5, \
             open(f'{folder}/cc_n_spectrum_diff_{threshold}.txt') as f6:
            conductance = f1.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            og_spectrum_diff = f2.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            v_spectrum_diff = f3.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            n_spectrum_diff = f4.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            cv_spectrum_diff = f5.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            cn_spectrum_diff = f6.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
        
        conductance = [float(i) for i in conductance]
        og_spectrum_diff = [float(i) for i in og_spectrum_diff]
        v_spectrum_diff = [float(i) for i in v_spectrum_diff]
        n_spectrum_diff = [float(i) for i in n_spectrum_diff]
        cv_spectrum_diff = [float(i) for i in cv_spectrum_diff]
        cn_spectrum_diff = [float(i) for i in cn_spectrum_diff]

        conductance_list.append(conductance[0])
        og_spectrum_diff_list.append(og_spectrum_diff[0])
        v_spectrum_diff_list.append(v_spectrum_diff[edge_removal-1])
        n_spectrum_diff_list.append(n_spectrum_diff[edge_removal-1])
        cv_spectrum_diff_list.append(cv_spectrum_diff[edge_removal-1])
        cn_spectrum_diff_list.append(cn_spectrum_diff[edge_removal-1])

        # Find lowest spectrum diff for voting and neighborhood
        lowest_spectrum_diff = math.inf
        lowest_spectrum_treshold = ''
        lowest_spectrum_algorithm = ''
        lowest_spectrum_index = 0

        for i in range (2, 4):
            with open(f'{folder}/cc_spectrum_diff_0.{i}.txt') as f5, \
                 open(f'{folder}/cc_n_spectrum_diff_0.{i}.txt') as f6:
                v_spectrum_diff = f5.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
                n_spectrum_diff = f6.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            
            v_spectrum_diff = [float(i) for i in v_spectrum_diff]
            n_spectrum_diff = [float(i) for i in n_spectrum_diff]

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

        with open(f'{folder}/cc_{lowest_spectrum_algorithm}spectrum_diff_{lowest_spectrum_treshold}.txt') as f7:
            lowest_spectrum_spectrum_diff = f7.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')

        lowest_spectrum_spectrum_diff = [float(i) for i in lowest_spectrum_spectrum_diff]

        lowest_spectrum_spectrum_diff_list.append(lowest_spectrum_spectrum_diff[lowest_spectrum_index])

    v_conductance_list = conductance_list.copy()
    n_conductance_list = conductance_list.copy()

    for i in range(len(v_spectrum_diff_list)-1, -1, -1):
        if v_spectrum_diff_list[i] == 999999:
            v_spectrum_diff_list.pop(i)
            v_conductance_list.pop(i)
        if n_spectrum_diff_list[i] == 999999:
            n_spectrum_diff_list.pop(i)
            n_conductance_list.pop(i)

    plt.plot(conductance_list, og_spectrum_diff_list, marker='o', label='Original')
    plt.plot(v_conductance_list, v_spectrum_diff_list, marker='o',label=f'Voting {edge_removal*10}% edge removal and threshold {threshold}')
    plt.plot(n_conductance_list, n_spectrum_diff_list, marker='o',label=f'Neighborhood {edge_removal*10}% edge removal and threshold {threshold}')
    plt.plot(conductance_list, cv_spectrum_diff_list, marker='o',label=f'CC Voting {edge_removal*10}% edge removal threshold {threshold}')
    plt.plot(conductance_list, cn_spectrum_diff_list, marker='o',label=f'CC Neighborhood {edge_removal*10}% edge removal threshold {threshold}')
    plt.plot(conductance_list, lowest_spectrum_spectrum_diff_list, marker='o',label='Lowest Spectrum')
    plt.xlabel('Conductance')
    plt.ylabel('Spectrum Difference')
    plt.title(f'Spectrum Difference vs Conductance for {graph} with |V|/|V_Q|={per}')
    plt.legend()
    plt.show()
