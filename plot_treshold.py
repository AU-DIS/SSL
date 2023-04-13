import sys
import matplotlib.pyplot as plt

graph = sys.argv[1]
per = '0.1'

if len(sys.argv) >= 3:
    per = sys.argv[2]
    
def plot_balanced_acc(graph, per, axs):
    with open(f'experiments/{graph}/{per}/OG/conductance.txt') as f1, \
         open(f'experiments/{graph}/{per}/OG/og_balanced_accuracy.txt') as f2, \
         open(f'experiments/{graph}/{per}/balanced_accuracy_{0.1}.txt') as f3, \
         open(f'experiments/{graph}/{per}/balanced_accuracy_{0.2}.txt') as f4, \
         open(f'experiments/{graph}/{per}/balanced_accuracy_{0.3}.txt') as f5, \
         open(f'experiments/{graph}/{per}/balanced_accuracy_{0.4}.txt') as f6, \
         open(f'experiments/{graph}/{per}/balanced_accuracy_{0.5}.txt') as f7, \
         open(f'experiments/{graph}/{per}/balanced_accuracy_{0.6}.txt') as f8:
        conductance = f1.read()
        conductance = conductance.replace('[', '')
        conductance = conductance.replace(']', '')   
        og_balanced_accuracy = f2.read()
        og_balanced_accuracy = og_balanced_accuracy.replace('[', '')
        og_balanced_accuracy = og_balanced_accuracy.replace(']', '')
        balanced_accuracy_1 = f3.read()
        balanced_accuracy_1 = balanced_accuracy_1.replace('][', ', ')
        balanced_accuracy_1 = balanced_accuracy_1.replace('[', '')
        balanced_accuracy_1 = balanced_accuracy_1.replace(']', '')
        balanced_accuracy_2 = f4.read()
        balanced_accuracy_2 = balanced_accuracy_2.replace('][', ', ')
        balanced_accuracy_2 = balanced_accuracy_2.replace('[', '')
        balanced_accuracy_2 = balanced_accuracy_2.replace(']', '')
        balanced_accuracy_3 = f5.read()
        balanced_accuracy_3 = balanced_accuracy_3.replace('][', ', ')
        balanced_accuracy_3 = balanced_accuracy_3.replace('[', '')
        balanced_accuracy_3 = balanced_accuracy_3.replace(']', '')
        balanced_accuracy_4 = f6.read()
        balanced_accuracy_4 = balanced_accuracy_4.replace('][', ', ')
        balanced_accuracy_4 = balanced_accuracy_4.replace('[', '')
        balanced_accuracy_4 = balanced_accuracy_4.replace(']', '')
        balanced_accuracy_5 = f7.read()
        balanced_accuracy_5 = balanced_accuracy_5.replace('][', ', ')
        balanced_accuracy_5 = balanced_accuracy_5.replace('[', '')
        balanced_accuracy_5 = balanced_accuracy_5.replace(']', '')
        balanced_accuracy_6 = f8.read()
        balanced_accuracy_6 = balanced_accuracy_6.replace('][', ', ')
        balanced_accuracy_6 = balanced_accuracy_6.replace('[', '')
        balanced_accuracy_6 = balanced_accuracy_6.replace(']', '')       

    conductance = conductance.split(", ")
    conductance = [float(i) for i in conductance]

    og_balanced_accuracy = og_balanced_accuracy.split(", ")
    og_balanced_accuracy = [float(i) for i in og_balanced_accuracy]

    balanced_accuracy_1 = balanced_accuracy_1.split(", ")
    balanced_accuracy_1 = [float(i) for i in balanced_accuracy_1]
    balanced_accuracy_2 = balanced_accuracy_2.split(", ")
    balanced_accuracy_2 = [float(i) for i in balanced_accuracy_2]
    balanced_accuracy_3 = balanced_accuracy_3.split(", ")
    balanced_accuracy_3 = [float(i) for i in balanced_accuracy_3]
    balanced_accuracy_4 = balanced_accuracy_4.split(", ")
    balanced_accuracy_4 = [float(i) for i in balanced_accuracy_4]
    balanced_accuracy_5 = balanced_accuracy_5.split(", ")
    balanced_accuracy_5 = [float(i) for i in balanced_accuracy_5]
    balanced_accuracy_6 = balanced_accuracy_6.split(", ")
    balanced_accuracy_6 = [float(i) for i in balanced_accuracy_6]

    shortest_length = min(len(conductance), len(og_balanced_accuracy), len(balanced_accuracy_1))

    x_values = conductance[0:shortest_length]
    y1_values = og_balanced_accuracy[0:shortest_length]
    y2_values = balanced_accuracy_1[0:shortest_length]
    y3_values = balanced_accuracy_2[0:shortest_length]
    y4_values = balanced_accuracy_3[0:shortest_length]
    y5_values = balanced_accuracy_4[0:shortest_length]
    y6_values = balanced_accuracy_5[0:shortest_length]
    y7_values = balanced_accuracy_6[0:shortest_length]

    axs.plot(x_values, y1_values, color='blue', marker='o')
    axs.plot(x_values, y2_values, color='red', marker='o')
    axs.plot(x_values, y3_values, color='green', marker='o')
    axs.plot(x_values, y4_values, color='orange', marker='o')
    axs.plot(x_values, y5_values, color='purple', marker='o')
    axs.plot(x_values, y6_values, color='brown', marker='o')
    axs.plot(x_values, y7_values, color='pink', marker='o')

    axs.legend(['Standard', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])

    axs.set_xlabel('Conductance')
    axs.set_ylabel('Balanced Accuracy')

    return axs

def plot_f1(graph, per, axs):
    with open(f'experiments/{graph}/{per}/OG/conductance.txt') as f1, \
         open(f'experiments/{graph}/{per}/OG/og_f1.txt') as f2, \
         open(f'experiments/{graph}/{per}/f1_{0.1}.txt') as f3, \
         open(f'experiments/{graph}/{per}/f1_{0.2}.txt') as f4, \
         open(f'experiments/{graph}/{per}/f1_{0.3}.txt') as f5, \
         open(f'experiments/{graph}/{per}/f1_{0.4}.txt') as f6, \
         open(f'experiments/{graph}/{per}/f1_{0.5}.txt') as f7, \
         open(f'experiments/{graph}/{per}/f1_{0.6}.txt') as f8:
        conductance = f1.read()
        conductance = conductance.replace('[', '')
        conductance = conductance.replace(']', '')   
        og_f1 = f2.read()
        og_f1 = og_f1.replace('[', '')
        og_f1 = og_f1.replace(']', '')
        f1_1 = f3.read()
        f1_1 = f1_1.replace('][', ', ')
        f1_1 = f1_1.replace('[', '')
        f1_1 = f1_1.replace(']', '')
        f1_2 = f4.read()
        f1_2 = f1_2.replace('][', ', ')
        f1_2 = f1_2.replace('[', '')
        f1_2 = f1_2.replace(']', '')
        f1_3 = f5.read()
        f1_3 = f1_3.replace('][', ', ')
        f1_3 = f1_3.replace('[', '')
        f1_3 = f1_3.replace(']', '')
        f1_4 = f6.read()
        f1_4 = f1_4.replace('][', ', ')
        f1_4 = f1_4.replace('[', '')
        f1_4 = f1_4.replace(']', '')
        f1_5 = f7.read()
        f1_5 = f1_5.replace('][', ', ')
        f1_5 = f1_5.replace('[', '')
        f1_5 = f1_5.replace(']', '')
        f1_6 = f8.read()
        f1_6 = f1_6.replace('][', ', ')
        f1_6 = f1_6.replace('[', '')
        f1_6 = f1_6.replace(']', '')       

    conductance = conductance.split(", ")
    conductance = [float(i) for i in conductance]

    og_f1 = og_f1.split(", ")
    og_f1 = [float(i) for i in og_f1]

    f1_1 = f1_1.split(", ")
    f1_1 = [float(i) for i in f1_1]
    f1_2 = f1_2.split(", ")
    f1_2 = [float(i) for i in f1_2]
    f1_3 = f1_3.split(", ")
    f1_3 = [float(i) for i in f1_3]
    f1_4 = f1_4.split(", ")
    f1_4 = [float(i) for i in f1_4]
    f1_5 = f1_5.split(", ")
    f1_5 = [float(i) for i in f1_5]
    f1_6 = f1_6.split(", ")
    f1_6 = [float(i) for i in f1_6]

    shortest_length = min(len(conductance), len(og_f1), len(f1_1))

    x_values = conductance[0:shortest_length]
    y1_values = og_f1[0:shortest_length]
    y2_values = f1_1[0:shortest_length]
    y3_values = f1_2[0:shortest_length]
    y4_values = f1_3[0:shortest_length]
    y5_values = f1_4[0:shortest_length]
    y6_values = f1_5[0:shortest_length]
    y7_values = f1_6[0:shortest_length]

    axs.plot(x_values, y1_values, color='blue', marker='o')
    axs.plot(x_values, y2_values, color='red', marker='o')
    axs.plot(x_values, y3_values, color='green', marker='o')
    axs.plot(x_values, y4_values, color='orange', marker='o')
    axs.plot(x_values, y5_values, color='purple', marker='o')
    axs.plot(x_values, y6_values, color='brown', marker='o')
    axs.plot(x_values, y7_values, color='pink', marker='o')

    axs.legend(['Standard', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])

    axs.set_xlabel('Conductance')
    axs.set_ylabel('f1')

    return axs

if __name__ == '__main__':
    fig = plt.figure()
    fig.suptitle('Graph: ' + graph + ', Percentage: ' + per)
    gs = fig.add_gridspec(2, hspace=0.5)
    
    axs1, axs2 = gs.subplots()

    axs1.set_title('f1 for different conductances and different voting thresholds')

    axs1 = plot_balanced_acc(graph, per, axs1)
    axs2 = plot_f1(graph, per, axs2)

    plt.show()
