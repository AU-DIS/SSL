import sys
import matplotlib.pyplot as plt

graph = sys.argv[1]
per = '0.1'

if len(sys.argv) >= 3:
    per = sys.argv[2]

def plot_balanced_acc(graph, per, axs):
    with open(f'experiments/{graph}/{per}//edge_removal.txt') as f1, \
         open(f'experiments/{graph}/{per}//og_balanced_accuracy.txt') as f2, \
         open(f'experiments/{graph}/{per}//balanced_accuracy_0.3.txt') as f3, \
         open(f'experiments/{graph}/{per}//n_balanced_accuracy_0.3.txt') as f4:
        edge_removal = f1.read()
        edge_removal = edge_removal.replace('][', ', ')
        edge_removal = edge_removal.replace('[', '')
        edge_removal = edge_removal.replace(']', '')   
        og_balanced_accuracy = f2.read()
        og_balanced_accuracy = og_balanced_accuracy.replace('[', '')
        og_balanced_accuracy = og_balanced_accuracy.replace(']', '')
        balanced_accuracy = f3.read()
        balanced_accuracy = balanced_accuracy.replace('][', ', ')
        balanced_accuracy = balanced_accuracy.replace('[', '')
        balanced_accuracy = balanced_accuracy.replace(']', '')
        n_balanced_accuracy = f4.read()
        n_balanced_accuracy = n_balanced_accuracy.replace('][', ', ')
        n_balanced_accuracy = n_balanced_accuracy.replace('[', '')
        n_balanced_accuracy = n_balanced_accuracy.replace(']', '')

    edge_removal = edge_removal.split(", ")
    edge_removal = [float(i) for i in edge_removal]

    og_balanced_accuracy = og_balanced_accuracy.split(", ")
    og_balanced_accuracy = [float(i) for i in og_balanced_accuracy]

    balanced_accuracy = balanced_accuracy.split(", ")
    balanced_accuracy = [float(i) for i in balanced_accuracy]

    n_balanced_accuracy = n_balanced_accuracy.split(", ")
    n_balanced_accuracy = [float(i) for i in n_balanced_accuracy]

    shortest_length = min(len(edge_removal), len(balanced_accuracy), len(n_balanced_accuracy))

    for i in range(len(og_balanced_accuracy), shortest_length):
        og_balanced_accuracy = og_balanced_accuracy + [og_balanced_accuracy[-1]]

    x_values = edge_removal[0:shortest_length]
    y1_values = og_balanced_accuracy[0:shortest_length]
    y2_values = balanced_accuracy[0:shortest_length]
    y3_values = n_balanced_accuracy[0:shortest_length]

    axs.plot(x_values, y1_values, color='blue', marker='o')
    axs.plot(x_values, y2_values, color='red', marker='o')
    axs.plot(x_values, y3_values, color='green', marker='o')

    axs.legend(['Standard', 'Voting', 'Neighborhood'])

    axs.set_xlabel('edge_removal')
    axs.set_ylabel('Balanced Accuracy')

    return axs

def plot_f1(graph, per, axs):
    with open(f'experiments/{graph}/{per}//edge_removal.txt') as f1, \
         open(f'experiments/{graph}/{per}//og_f1.txt') as f2, \
         open(f'experiments/{graph}/{per}//f1_0.3.txt') as f3, \
         open(f'experiments/{graph}/{per}//n_f1_0.3.txt') as f4:
        edge_removal = f1.read()
        edge_removal = edge_removal.replace('][', ', ')
        edge_removal = edge_removal.replace('[', '')
        edge_removal = edge_removal.replace(']', '')   
        og_f1 = f2.read()
        og_f1 = og_f1.replace('[', '')
        og_f1 = og_f1.replace(']', '')
        _f1 = f3.read()
        _f1 = _f1.replace('][', ', ')
        _f1 = _f1.replace('[', '')
        _f1 = _f1.replace(']', '')
        n_f1 = f4.read()
        n_f1 = n_f1.replace('][', ', ')
        n_f1 = n_f1.replace('[', '')
        n_f1 = n_f1.replace(']', '')

    edge_removal = edge_removal.split(", ")
    edge_removal = [float(i) for i in edge_removal]

    og_f1 = og_f1.split(", ")
    og_f1 = [float(i) for i in og_f1]

    _f1 = _f1.split(", ")
    _f1 = [float(i) for i in _f1]

    n_f1 = n_f1.split(", ")
    n_f1 = [float(i) for i in n_f1]

    shortest_length = min(len(edge_removal), len(_f1), len(n_f1))
    
    for i in range(len(og_f1), shortest_length):
        og_f1 = og_f1 + [og_f1[-1]]

    x_values = edge_removal[0:shortest_length]
    y1_values = og_f1[0:shortest_length]
    y2_values = _f1[0:shortest_length]
    y3_values = n_f1[0:shortest_length]

    axs.plot(x_values, y1_values, color='blue', marker='o')
    axs.plot(x_values, y2_values, color='red', marker='o')
    axs.plot(x_values, y3_values, color='green', marker='o')

    axs.legend(['Standard', 'Voting', 'Neighborhood'])

    axs.set_xlabel('edge_removal')
    axs.set_ylabel('f1')

if __name__ == '__main__':    
    fig = plt.figure()

    with open(f'experiments/{graph}/{per}/conductance.txt') as f1:
        conductance = f1.read()
        conductance = conductance.replace('][', ', ')
        conductance = conductance.replace('[', '')
        conductance = conductance.replace(']', '')
    conductance = conductance.split(", ")
    conductance = conductance[0]
    conductance = conductance[0: 4]

    fig.suptitle('Graph: ' + graph + ', Percentage: ' + per + ', Conductance: ' + conductance)
    gs = fig.add_gridspec(2, hspace=0.5)

    axs1, axs2 = gs.subplots()

    axs1.set_title('Balanced Accuracy and f1 for different edge_removals')

    axs1 = plot_balanced_acc(graph, per, axs1)
    axs2 = plot_f1(graph, per, axs2)

    plt.show()
