import sys
import matplotlib.pyplot as plt

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

def plot_balanced_acc(graph, per, axs):
    with open(f'experiments_edge_removal/{graph}/{per}/{folder}/edge_removal.txt') as f1, \
         open(f'experiments_edge_removal/{graph}/{per}/{folder}/{algorithm}balanced_accuracy_{threshold}.txt') as f2:
        edge_removal = f1.read()
        edge_removal = edge_removal.replace('][', ', ')
        edge_removal = edge_removal.replace('[', '')
        edge_removal = edge_removal.replace(']', '')   
        balanced_accuracy = f2.read()
        balanced_accuracy = balanced_accuracy.replace('][', ', ')
        balanced_accuracy = balanced_accuracy.replace('[', '')
        balanced_accuracy = balanced_accuracy.replace(']', '')
        
    edge_removal = edge_removal.split(", ")
    edge_removal = [float(i) for i in edge_removal]

    balanced_accuracy = balanced_accuracy.split(", ")
    balanced_accuracy = [float(i) for i in balanced_accuracy]

    shortest_length = min(len(edge_removal), len(balanced_accuracy))

    x_values = edge_removal[0:shortest_length]
    y_values = balanced_accuracy[0:shortest_length]

    axs.plot(x_values, y_values, color='blue', marker='o')

    axs.set_xlabel('Edge removal')
    axs.set_ylabel('Balanced accuracy')

    return axs

def plot_spectrum_diff(graph, per, axs):
    with open(f'experiments_edge_removal/{graph}/{per}/{folder}/edge_removal.txt') as f1, \
         open(f'experiments_edge_removal/{graph}/{per}/{folder}/{algorithm}spectrum_diff_{threshold}.txt') as f3:
        edge_removal = f1.read()
        edge_removal = edge_removal.replace('][', ', ')
        edge_removal = edge_removal.replace('[', '')
        edge_removal = edge_removal.replace(']', '')   
        spectrum_diff = f3.read()
        spectrum_diff = spectrum_diff.replace('][', ', ')
        spectrum_diff = spectrum_diff.replace('[', '')
        spectrum_diff = spectrum_diff.replace(']', '')
        
    edge_removal = edge_removal.split(", ")
    edge_removal = [float(i) for i in edge_removal]

    spectrum_diff = spectrum_diff.split(", ")
    spectrum_diff = [float(i) for i in spectrum_diff]

    shortest_length = min(len(edge_removal), len(spectrum_diff))

    x_values = edge_removal[0:shortest_length]
    y2_values = spectrum_diff[0:shortest_length]

    axs.plot(x_values, y2_values, color='red', marker='o')

    axs.invert_yaxis()

    axs.set_xlabel('Edge removal')
    axs.set_ylabel('Spectrum difference')

    return axs

if __name__ == '__main__':    
    fig = plt.figure()

    with open(f'experiments_edge_removal/{graph}/{per}/{folder}/conductance.txt') as f1:
        conductance = f1.read()
        conductance = conductance.replace('][', ', ')
        conductance = conductance.replace('[', '')
        conductance = conductance.replace(']', '')
    conductance = conductance.split(", ")
    conductance = conductance[0]

    fig.suptitle('Graph: ' + graph + ', Percentage: ' + per + ', Conductance: ' + conductance)
    gs = fig.add_gridspec(2, hspace=0.5)

    axs1, axs2 = gs.subplots()

    axs1.set_title('Balanced Accuracy and f1 for different edge_removals')

    axs1 = plot_balanced_acc(graph, per, axs1)
    axs2 = plot_spectrum_diff(graph, per, axs2)

    plt.show()
