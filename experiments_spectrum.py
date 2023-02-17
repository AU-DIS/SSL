import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import csv

def edgelist_to_adjmatrix(edgeList_file):
    edge_list = np.loadtxt(edgeList_file, usecols=range(2))

    n = int(np.amax(edge_list) + 1)
    e = np.shape(edge_list)[0]
    a = np.zeros((n, n))

    # make adjacency matrix A1
    for i in range(0, e):
        n1 = int(edge_list[i, 0])  # - 1
        n2 = int(edge_list[i, 1])  # - 1
        a[n1, n2] = 1.0
        a[n2, n1] = 1.0

    return a



def diff_of_spectrums(edgefile, part_file):
    A1=edgelist_to_adjmatrix(edgefile)
    G=nx.from_numpy_matrix(A1)
    A = nx.to_numpy_matrix(G)

    D = np.diagonal(np.sum(A, axis=1)*np.ones(G.number_of_nodes()).reshape(1,G.number_of_nodes())*np.identity(G.number_of_nodes()))*np.ones(G.number_of_nodes()).reshape(1,G.number_of_nodes())*np.identity(G.number_of_nodes())
    print((D**(-0.5)))
    L = D - A
    print(L)
    D12 = (D**(-0.5))
    for i in range(D12.shape[0]):
        for j in range(D12.shape[1]):
            if D12[i,j] > 50000000:
                D12[i,j] = 0
    NormalizedL = D12*L*D12
    init_spectrum = np.linalg.eig(NormalizedL)
    init_spectrum = (list(init_spectrum))[0]
    init_spectrum.sort()
    init_spectrum = (init_spectrum[1:]-np.min(init_spectrum[1:]))/np.max(init_spectrum)
    pos = [int(i*(len(init_spectrum)-1)/10) for i in range(11)]
    print(pos, len(pos))
    init_spectrum = init_spectrum[pos]
    print(init_spectrum)
    return init_spectrum


if __name__ == '__main__':
    graph_names = ['ant', 'football', 'highschool', 'malaria', 'powerlaw_200_50_50', 'renyi_200_50', 'barabasi_200_50']
    colors = ['red', 'blue', 'green', 'orange', 'brown', 'yellow', 'black']
    plotdiff = {}
    for graph_name in graph_names:
        folder_no = 0
        per = 0.3
        ext = str(int(per*100))
        part_file = './data/'+graph_name+'/'+str(int(per*100))+'/'+str(folder_no)+'/'+graph_name+'_'+ext+'_nodes.txt'
        lr = 1.0
        ext = str(int(per*100))+"_"+str(int(lr*100))
        edgefile = './data/'+graph_name+'/'+str(int(per*100))+'/'+str(folder_no)+'/'+graph_name+'_'+ext+'.txt'
        plotdiff[graph_name] = diff_of_spectrums(edgefile, part_file)
    for i in range(len(graph_names)):
        plt.plot( plotdiff[graph_names[i]], color = colors[i])
    plt.show()
    with open(f'./csv_results/spectrum_of_graphs.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file, delimiter=" ")
        writer.writerow(['ant', 'football', 'highschool', 'malaria', 'powerlaw_200_50_50', 'renyi_200_50', 'barabasi_200_50','x'])
        counter = 0
        for i in range(11):
            new_line = [plotdiff[graph_name][i] for graph_name in graph_names]
            new_line.append(int((i)*10))
            writer.writerow(new_line) 
        
    