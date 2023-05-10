import math
import sys
import matplotlib.pyplot as plt
import numpy as np

graph = 'football'
per = '0.1'
folder = '2'
threshold = '0.3'
algorithm = 'n_'

if len(sys.argv) >= 2:
    graph = sys.argv[1]
if len(sys.argv) >= 3:
    per = sys.argv[2]
if len(sys.argv) >= 4:
    folder = sys.argv[3]
if len(sys.argv) >= 5:
    threshold = sys.argv[4]
if len(sys.argv) >= 6: 
    if sys.argv[5] == 'voting':
        algorithm = ''

def plot_correlation(graph, per):
    with open(f'experiments_edge_removal/{graph}/{per}/{folder}/edge_removal.txt') as f1, \
         open(f'experiments_edge_removal/{graph}/{per}/{folder}/{algorithm}balanced_accuracy_{threshold}.txt') as f2, \
         open(f'experiments_edge_removal/{graph}/{per}/{folder}/{algorithm}spectrum_diff_{threshold}.txt') as f3:
        edge_removal = f1.read()
        edge_removal = edge_removal.replace('][', ', ')
        edge_removal = edge_removal.replace('[', '')
        edge_removal = edge_removal.replace(']', '')   
        balanced_accuracy = f2.read()
        balanced_accuracy = balanced_accuracy.replace('][', ', ')
        balanced_accuracy = balanced_accuracy.replace('[', '')
        balanced_accuracy = balanced_accuracy.replace(']', '')
        spectrum_diff = f3.read()
        spectrum_diff = spectrum_diff.replace('][', ', ')
        spectrum_diff = spectrum_diff.replace('[', '')
        spectrum_diff = spectrum_diff.replace(']', '')
        
    edge_removal = edge_removal.split(", ")
    edge_removal = [float(i) for i in edge_removal]

    balanced_accuracy = balanced_accuracy.split(", ")
    balanced_accuracy = [float(i) for i in balanced_accuracy]

    spectrum_diff = spectrum_diff.split(", ")
    spectrum_diff = [float(i) for i in spectrum_diff]

    for i in range(len(spectrum_diff)-1, -1, -1):
        if spectrum_diff[i] == math.inf:
            del spectrum_diff[i]
            del balanced_accuracy[i]
            del edge_removal[i]

    shortest_length = min(len(edge_removal), len(balanced_accuracy), len(spectrum_diff))

    x_values = balanced_accuracy[0:shortest_length]
    y_values = spectrum_diff[0:shortest_length]

    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)

    plt.scatter(x_values, y_values)

    z = np.polyfit(x_values, y_values, 1)
    p = np.poly1d(z)

    plt.plot(x_values, p(x_values),"--k")
    plt.xlabel('balanced accuracy')
    plt.ylabel('spectral diff')
    myalg = algorithm
    print(myalg)
    if myalg == '':
        myalg = 'voting'
    else:
        myalg = 'neighborhood'
    plt.title(f"Correlation graph for {graph} with V/VQ = {per} and with {myalg}")
    plt.show()

    return

if __name__ == '__main__':    
    plot_correlation(graph, per)
