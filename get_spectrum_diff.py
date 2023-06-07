import sys
import os
from problem.spectral_subgraph_localization import find_voting_majority, edgelist_to_adjmatrix
from experiments_ssl import solution_graph, graph_edit_distance, use_graph_edit_distance_generator, enforce_cardinality_constraint_by_spectrum, spectrum_from_graph, spectrum_abs_diff, spectrum_square_diff
from problem.dijkstra import DijkstraSolution
import torch
import networkx as nx

graph = 'football'
per = '0.1'

if len(sys.argv) >= 2:
    graph = sys.argv[1]
if len(sys.argv) >= 3:
    per = sys.argv[2]

if __name__ == '__main__':
    rootdir = f'experiments_final/{graph}/{per}'
    directories = []

    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            directories.append(d)

    directories.sort()

    for folder in directories:
        with open(f'{folder}/votes') as votes_file, \
                open(f'{folder}/ground_truth') as gt_file: 
                    *_, per, lr = folder.split('/')
                    per = float(per)
                    lr = float(lr)

                    ext = str(int(per*100))+"_"+str(int(lr))
                    edgefile = './data/'+graph+'/'+str(int(per*100))+'/1/'+graph+'_'+ext+'.txt'
                    A1=edgelist_to_adjmatrix(edgefile)
                    G=nx.from_numpy_matrix(A1)
                    A = torch.tensor(nx.to_numpy_matrix(G))

                    gt_string = gt_file.read().replace('])','').replace('\n','')
                    gt = gt_string.split('][')[0]
                    gt = gt.replace('[','').replace(']','').replace('array(','').split(', ')
                    gt = [float(x) for x in gt]
                    gt_indices = [i for i, res in enumerate(gt) if res == 0]
                    _G = G.copy()
                    Q = _G.subgraph(gt_indices)
                    ref_spectrum = spectrum_from_graph(Q)
                    length_of_query = len(Q.nodes())

                    thresholds = [0.2, 0.3, 0.4]
                    votes_string = votes_file.read().replace('])', '').replace('\n', '')
                    votes_list = []
                    for tensor in votes_string.split(']['):
                        tensor = tensor.replace('[','').replace(']','').replace('tensor(','')
                        votes_list.append(tensor.split(', '))

                    for votes in votes_list:
                        votes = [float(x) for x in votes]
                        votes = torch.tensor(votes)
                        experiments_to_make = int(torch.max(votes))

                        # Find solutions for standard voting 
                        for threshold in thresholds:
                            v = find_voting_majority(votes, experiments_to_make, threshold)
                            S = solution_graph(G, v)
                            spectrum = spectrum_from_graph(S)
                            spectrum_diff = spectrum_square_diff(ref_spectrum, spectrum)

                            # Write it!
                            f = open(f"{folder}/spectrum_diff2_{threshold}.txt", "a+")
                            f.write(str([spectrum_diff]))

                            # Do the same for cardiality constraint!
                            v = enforce_cardinality_constraint_by_spectrum(G, v, ref_spectrum)
                            S = solution_graph(G, v)
                            spectrum = spectrum_from_graph(S)
                            spectrum_diff = spectrum_square_diff(ref_spectrum, spectrum)

                            # Write it!
                            f = open(f"{folder}/cc_spectrum_diff2_{threshold}.txt", "a+")
                            f.write(str([spectrum_diff]))

                        # Find solutions for neighborhood 
                        for threshold in thresholds:
                            dijkstra = DijkstraSolution(A, votes, experiments_to_make, "cubic", threshold, "constant", length_of_query)
                            v = dijkstra.solution()
                            S = solution_graph(G, v)
                            spectrum = spectrum_from_graph(S)
                            spectrum_diff = spectrum_square_diff(ref_spectrum, spectrum)

                            # Write it!
                            f = open(f"{folder}/n_spectrum_diff2_{threshold}.txt", "a+")
                            f.write(str([spectrum_diff]))

                            # Do the same for cardiality constraint!
                            v = enforce_cardinality_constraint_by_spectrum(G, v, ref_spectrum)
                            S = solution_graph(G, v)
                            spectrum = spectrum_from_graph(S)
                            spectrum_diff = spectrum_square_diff(ref_spectrum, spectrum)

                            # Write it!
                            f = open(f"{folder}/cc_n_spectrum_diff2_{threshold}.txt", "a+")
                            f.write(str([spectrum_diff]))
