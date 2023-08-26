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
    rootdir = f'CONE/experiments{suffix}/{graph}/{per}'
    
    with open(f'{rootdir}/conductances.txt') as f1, \
         open(f'{rootdir}/balanced_accuracies.txt') as f2, \
         open(f'{rootdir}/spectrum_diff.txt') as f3:
        conductance = f1.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
        balanced_accuracies = f2.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
        spectrum_diff = f3.read().replace('][', ', ').replace('[', '').replace(']', '').split(', ')
     
        conductance = [float(i) for i in conductance]
        balanced_accuracies = [float(i) for i in balanced_accuracies]
        spectrum_diff = [float(i) for i in spectrum_diff]

    return conductance, balanced_accuracies, spectrum_diff

def plot(plt, conductance_list, balanced_accuracy_list, label, should_scatter = False):

    combined = zip(conductance_list, balanced_accuracy_list)
    sorted_combined = sorted(combined, key=lambda x: x[0])
    x, y = zip(*sorted_combined)

    x = np.array(x)
    y = np.array(y)

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    residuals = y - p(x)
    residual_variance = np.var(residuals)

    print(f"Variance of residuals for {label}:", residual_variance)
    #print(f'{np.round(z[0],4)}*x^2 + {np.round(z[1],4)}*x + {np.round(z[2],4)}')

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

    return np.sqrt(conductance_list), balanced_accuracy_list


def entry_averaging():
    con1, ba1, sd1 = get_data_from_folder("1")
    con2, ba2, sd2 = get_data_from_folder("2")
    con3, ba3, sd3 = get_data_from_folder("3")
    con4, ba4, sd4 = get_data_from_folder("4")
    con5, ba5, sd5 = get_data_from_folder("5")

    _conductance_list = list(zip(con1, con2, con3, con4, con5))
    conductance_list =  [mean(t) for t in _conductance_list]
            
    _ba = list(zip(ba1, ba2, ba3, ba4, ba5))
    balanced_accuracy_list =([mean((t)) for t in _ba])

    _sd = list(zip(sd1, sd2, sd3, sd4, sd5))
    spectral_diff_list =([mean(np.sqrt(t)) for t in _sd])
                                                                                                                                
                                                                                                                                    
    with open(f'ssl_graphs/{graph}_{per}_CONE.csv', 'w', encoding='UTF8', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=" ") 
        writer.writerow(['conductance','ba', 'sd'])
        for i in range(len(conductance_list)):
            writer.writerow([conductance_list[i], balanced_accuracy_list[i], spectral_diff_list[i]])



def regression():
    conductance_list, balanced_accuracy_list = get_data_from_all_folders()

    plot(plt, conductance_list, balanced_accuracy_list, "CONE")

    plt.xlabel('Conductance')
    plt.ylabel('Balanced Accuracy')
    plt.title(f'CONE - Balaned accuracy vs Conductance for {graph} with |V|/|V_Q|={per}')

if __name__ == '__main__':
   # regression()
    entry_averaging()

