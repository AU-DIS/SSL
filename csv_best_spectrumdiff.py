import math
import sys
import os
import csv

import numpy as np
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
    rootdir = f'experiments_final/{graph}/{per}'
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
    cv_balanced_accuracy_list = []
    cn_balanced_accuracy_list = []
    lowest_spectrum_balanced_accuracy_list = []

    for folder in directories:
        with open(f'{folder}/conductance.txt') as f1, \
             open(f'{folder}/og_balanced_accuracy.txt') as f2, \
             open(f'{folder}/balanced_accuracy_{threshold}.txt') as f3, \
             open(f'{folder}/n_balanced_accuracy_{threshold}.txt') as f4, \
             open(f'{folder}/cc_balanced_accuracy_{threshold}.txt') as f5, \
             open(f'{folder}/cc_n_balanced_accuracy_{threshold}.txt') as f6:
            conductance = f1.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            og_balanced_accuracy = f2.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            v_balanced_accuracy = f3.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            n_balanced_accuracy = f4.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            cv_balanced_accuracy = f5.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            cn_balanced_accuracy = f6.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
        
        conductance = [float(i) for i in conductance]
        og_balanced_accuracy = [float(i) for i in og_balanced_accuracy]
        v_balanced_accuracy = [float(i) for i in v_balanced_accuracy]
        n_balanced_accuracy = [float(i) for i in n_balanced_accuracy]
        cv_balanced_accuracy = [float(i) for i in cv_balanced_accuracy]
        cn_balanced_accuracy = [float(i) for i in cn_balanced_accuracy]

        conductance_list.append(conductance[0])
        og_balanced_accuracy_list.append(og_balanced_accuracy[0])
        v_balanced_accuracy_list.append(v_balanced_accuracy[edge_removal-1])
        n_balanced_accuracy_list.append(n_balanced_accuracy[edge_removal-1])
        cv_balanced_accuracy_list.append(cv_balanced_accuracy[edge_removal-1])
        cn_balanced_accuracy_list.append(cn_balanced_accuracy[edge_removal-1])

        # Find lowest spectrum diff for voting and neighborhood
        lowest_spectrum_diff = math.inf
        lowest_spectrum_treshold = ''
        lowest_spectrum_algorithm = ''
        lowest_spectrum_index = 0

        for i in range (1, 7):
            with open(f'{folder}/cc_spectrum_diff_0.{i}.txt') as f5, \
                 open(f'{folder}/cc_n_spectrum_diff_0.{i}.txt') as f6:
                v_spectrum_diff = f5.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
                n_spectrum_diff = f6.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
            
            v_spectrum_diff = [float(i) for i in v_spectrum_diff]
            n_spectrum_diff = [float(i) for i in n_spectrum_diff]

            for j in range(len(v_spectrum_diff)):
                if v_spectrum_diff[j] < lowest_spectrum_diff:
                    lowest_spectrum_diff = v_spectrum_diff[j]
                    lowest_spectrum_treshold = f'0.{i}'
                    lowest_spectrum_index = j
                    lowest_spectrum_algorithm = ''
                if n_spectrum_diff[j] < lowest_spectrum_diff:
                    lowest_spectrum_diff = n_spectrum_diff[j]
                    lowest_spectrum_treshold = f'0.{i}'
                    lowest_spectrum_index = j
                    lowest_spectrum_algorithm = 'n_'

        with open(f'{folder}/cc_{lowest_spectrum_algorithm}balanced_accuracy_{lowest_spectrum_treshold}.txt') as f7:
            lowest_spectrum_balanced_accuracy = f7.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')

        lowest_spectrum_balanced_accuracy = [float(i) for i in lowest_spectrum_balanced_accuracy]

        lowest_spectrum_balanced_accuracy_list.append(lowest_spectrum_balanced_accuracy[lowest_spectrum_index])

    conductance_list = np.array(conductance_list)
    og_balanced_accuracy_list = np.array(og_balanced_accuracy_list)
    v_balanced_accuracy_list = np.array(v_balanced_accuracy_list)
    n_balanced_accuracy_list = np.array(n_balanced_accuracy_list)
    cv_balanced_accuracy_list = np.array(cv_balanced_accuracy_list)
    cn_balanced_accuracy_list = np.array(cn_balanced_accuracy_list)
    lowest_spectrum_balanced_accuracy_list = np.array(lowest_spectrum_balanced_accuracy_list)

    #Transpose the arrays and combine them into one matrix
    data = np.array([conductance_list, 
                     og_balanced_accuracy_list, 
                     v_balanced_accuracy_list, 
                     n_balanced_accuracy_list, 
                     cv_balanced_accuracy_list, 
                     cn_balanced_accuracy_list, 
                     lowest_spectrum_balanced_accuracy_list]).T


    labels = ['conductance', 'ssl', 'voting', 'neighborhood', 'cc_voting', 'cc_neighborhood', 'lowest_spectrum']
    
    with open('best_spectrum.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(labels)

        # write multiple rows
        writer.writerows(data)