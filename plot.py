import sys
import matplotlib.pyplot as plt

graph = sys.argv[1]
per = sys.argv[2]
voting_threshold = sys.argv[3]

def plot_balanced_acc(graph, per, voting_threshold, axs):
    with open(f'experiments/{graph}/{per}/conductance.txt') as f1, \
         open(f'experiments/{graph}/{per}/og_balanced_accuracy.txt') as f2, \
         open(f'experiments/{graph}/{per}/balanced_accuracy_{voting_threshold}.txt') as f3, \
         open(f'experiments/{graph}/{per}/n_balanced_accuracy_{voting_threshold}.txt') as f4:
        conductance = f1.read()
        conductance = conductance.replace('[', '')
        conductance = conductance.replace(']', '')   
        og_balanced_accuracy = f2.read()
        og_balanced_accuracy = og_balanced_accuracy.replace('[', '')
        og_balanced_accuracy = og_balanced_accuracy.replace(']', '')
        balanced_accuracy = f3.read()
        balanced_accuracy = balanced_accuracy.replace('[', '')
        balanced_accuracy = balanced_accuracy.replace(']', '')
        n_balanced_accuracy = f4.read()
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

    # Define two arrays of data to plot
    # TODO Conductance kommer ikke i stigende rækkefølge pga. den måde vi kører eksperimenterne på. Derfor fake dataset...
    x_values = conductance
    y1_values = og_balanced_accuracy
    y2_values = balanced_accuracy
    y3_values = n_balanced_accuracy

    # Create a new figure and axis object

    # Plot the first set of data as a blue line
    axs[0].plot(x_values, y1_values, color='blue')

    # Plot the second set of data as a red line
    axs[0].plot(x_values, y2_values, color='red')

    axs[0].plot(x_values, y3_values, color='green')

    # Add a legend to the plot
    axs[0].legend(['Standard', 'Voting', 'Neighborhood'])

    # Set the x and y axis labels
    axs[0].set_xlabel('Conductance')
    axs[0].set_ylabel('Balanced Accuracy')

    # Set the title of the plot 

    # Display the plot
    return axs

def plot_f1(graph, per, voting_threshold, axs):
    with open(f'experiments/{graph}/{per}/conductance.txt') as f1, \
         open(f'experiments/{graph}/{per}/og_f1.txt') as f2, \
         open(f'experiments/{graph}/{per}/f1_{voting_threshold}.txt') as f3, \
         open(f'experiments/{graph}/{per}/n_f1_{voting_threshold}.txt') as f4:
        conductance = f1.read()
        conductance = conductance.replace('[', '')
        conductance = conductance.replace(']', '')   
        og_f1 = f2.read()
        og_f1 = og_f1.replace('[', '')
        og_f1 = og_f1.replace(']', '')
        _f1 = f3.read()
        _f1 = _f1.replace('[', '')
        _f1 = _f1.replace(']', '')
        n_f1 = f4.read()
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

    # Define two arrays of data to plot
    # TODO Conductance kommer ikke i stigende rækkefølge pga. den måde vi kører eksperimenterne på. Derfor fake dataset...
    x_values = conductance
    y1_values = og_f1
    y2_values = _f1
    y3_values = n_f1

    # Plot the first set of data as a blue line
    axs[1].plot(x_values, y1_values, color='blue')

    # Plot the second set of data as a red line
    axs[1].plot(x_values, y2_values, color='red')

    axs[1].plot(x_values, y3_values, color='green')

    # Add a legend to the plot
    axs[1].legend(['Standard', 'Voting', 'Neighborhood'])

    # Set the x and y axis labels
    axs[1].set_xlabel('Conductance')
    axs[1].set_ylabel('f1')

    # Set the title of the plot

    # Display the plot

if __name__ == '__main__':
    fig = plt.figure()
    fig.suptitle('Graph: ' + graph + ', Percentage: ' + per + ', Voting Threshold: ' + voting_threshold)
    gs = fig.add_gridspec(2, hspace=0.5)
    axs = gs.subplots()
    axs = plot_balanced_acc(graph, per, voting_threshold, axs)
    axs[0].set_title('Balanced Accuracy and f1 for different conductances')
    axs = plot_f1(graph, per, voting_threshold, axs)
    plt.show()
