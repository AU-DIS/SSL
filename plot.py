import sys
import matplotlib.pyplot as plt

graph = sys.argv[1]
per = '0.1'
voting_threshold = None

if len(sys.argv) >= 3:
    per = sys.argv[2]
    
if len(sys.argv) >= 4:
    voting_threshold = sys.argv[3]

def plot_balanced_acc(graph, per, voting_threshold, axs):
    with open(f'experiments/{graph}/{per}/OG/conductance.txt') as f1, \
         open(f'experiments/{graph}/{per}/OG/og_balanced_accuracy.txt') as f2, \
         open(f'experiments/{graph}/{per}/balanced_accuracy_{voting_threshold}.txt') as f3, \
         open(f'experiments/{graph}/{per}/n_balanced_accuracy_{voting_threshold}.txt') as f4:
        conductance = f1.read()
        conductance = conductance.replace('[', '')
        conductance = conductance.replace(']', '')   
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

    conductance = conductance.split(", ")
    conductance = [float(i) for i in conductance]

    og_balanced_accuracy = og_balanced_accuracy.split(", ")
    og_balanced_accuracy = [float(i) for i in og_balanced_accuracy]

    balanced_accuracy = balanced_accuracy.split(", ")
    balanced_accuracy = [float(i) for i in balanced_accuracy]

    n_balanced_accuracy = n_balanced_accuracy.split(", ")
    n_balanced_accuracy = [float(i) for i in n_balanced_accuracy]

    shortest_length = min(len(conductance), len(og_balanced_accuracy), len(balanced_accuracy), len(n_balanced_accuracy))

    x_values = conductance[0:shortest_length]
    y1_values = og_balanced_accuracy[0:shortest_length]
    y2_values = balanced_accuracy[0:shortest_length]
    y3_values = n_balanced_accuracy[0:shortest_length]

    axs.plot(x_values, y1_values, color='blue', marker='o')
    axs.plot(x_values, y2_values, color='red', marker='o')
    axs.plot(x_values, y3_values, color='green', marker='o')

    axs.legend(['Standard', 'Voting', 'Neighborhood'])

    axs.set_xlabel('Conductance')
    axs.set_ylabel('Balanced Accuracy')

    return axs

def plot_f1(graph, per, voting_threshold, axs):
    with open(f'experiments/{graph}/{per}/OG/conductance.txt') as f1, \
         open(f'experiments/{graph}/{per}/OG/og_f1.txt') as f2, \
         open(f'experiments/{graph}/{per}/f1_{voting_threshold}.txt') as f3, \
         open(f'experiments/{graph}/{per}/n_f1_{voting_threshold}.txt') as f4:
        conductance = f1.read()
        conductance = conductance.replace('[', '')
        conductance = conductance.replace(']', '')   
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

    conductance = conductance.split(", ")
    conductance = [float(i) for i in conductance]

    og_f1 = og_f1.split(", ")
    og_f1 = [float(i) for i in og_f1]

    _f1 = _f1.split(", ")
    _f1 = [float(i) for i in _f1]

    n_f1 = n_f1.split(", ")
    n_f1 = [float(i) for i in n_f1]

    shortest_length = min(len(conductance), len(og_f1), len(_f1), len(n_f1))

    x_values = conductance[0:shortest_length]
    y1_values = og_f1[0:shortest_length]
    y2_values = _f1[0:shortest_length]
    y3_values = n_f1[0:shortest_length]

    axs.plot(x_values, y1_values, color='blue', marker='o')
    axs.plot(x_values, y2_values, color='red', marker='o')
    axs.plot(x_values, y3_values, color='green', marker='o')

    axs.legend(['Standard', 'Voting', 'Neighborhood'])

    axs.set_xlabel('Conductance')
    axs.set_ylabel('f1')

def plot_all_thresholds(graph, per):
    fig = plt.figure()
    fig.suptitle('Graph: ' + graph + ', Percentage: ' + per + '\n Balanced Accuracy and f1 for different conductances over different voting thresholds')
    gs = fig.add_gridspec(2, 6, hspace=0.3, wspace=0.3)

    (axs1, axs2, axs3, axs4, axs5, axs6), (axs7, axs8, axs9, axs10, axs11, axs12) = gs.subplots(sharey='row')

    axs1 = plot_balanced_acc(graph, per, 0.1, axs1)
    axs2 = plot_balanced_acc(graph, per, 0.2, axs2)
    axs3 = plot_balanced_acc(graph, per, 0.3, axs3)
    axs4 = plot_balanced_acc(graph, per, 0.4, axs4)
    axs5 = plot_balanced_acc(graph, per, 0.5, axs5)
    axs6 = plot_balanced_acc(graph, per, 0.6, axs6)
    axs7 = plot_f1(graph, per, 0.1, axs7)
    axs8 = plot_f1(graph, per, 0.2, axs8)
    axs9 = plot_f1(graph, per, 0.3, axs9)
    axs10 = plot_f1(graph, per, 0.4, axs10)
    axs11 = plot_f1(graph, per, 0.5, axs11)
    axs12 = plot_f1(graph, per, 0.6, axs12)

    axs1.set_title('Voting Threshold: 0.1')
    axs2.set_title('Voting Threshold: 0.2')
    axs3.set_title('Voting Threshold: 0.3')
    axs4.set_title('Voting Threshold: 0.4')
    axs5.set_title('Voting Threshold: 0.5')
    axs6.set_title('Voting Threshold: 0.6')

    plt.show()


def plot_selected_thresholds(graph, per):
    fig = plt.figure()
    fig.suptitle('Graph: ' + graph + ', Percentage: ' + per + ', Voting Threshold: ' + voting_threshold)
    gs = fig.add_gridspec(2, hspace=0.5)

    axs1, axs2 = gs.subplots()

    axs1.set_title('Balanced Accuracy and f1 for different conductances')

    axs1 = plot_balanced_acc(graph, per, voting_threshold, axs1)
    axs2 = plot_f1(graph, per, voting_threshold, axs2)

    plt.show()

if __name__ == '__main__':    
    if voting_threshold == None:
        plot_all_thresholds(graph, per)
    else:
        plot_selected_thresholds(graph, per)
