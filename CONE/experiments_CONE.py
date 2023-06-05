import numpy as np
import sklearn.metrics.pairwise

import networkx as nx
try: import pickle as pickle
except ImportError:
    import pickle
import scipy.sparse as sps
import argparse
import time
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.neighbors import KDTree
import unsup_align
import embedding
from pathlib import Path
import os

#from functionalMaps_nn import *


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



def parse_args():
    parser = argparse.ArgumentParser(description="Run CONE Align.")

    parser.add_argument('--true_align', nargs='?', default='data/synthetic-combined/arenas/arenas950-1/arenas_edges-mapping-permutation.txt',
                        help='True alignment file.')
    parser.add_argument('--combined_graph', nargs='?', default='data/synthetic-combined/arenas/arenas950-1/arenas_combined_edges.txt', help='Edgelist of combined input graph.')
    parser.add_argument('--output_stats', nargs='?', default='output/stats/arenas/arenas950-1.log', help='Output path for log file.')
    parser.add_argument('--store_align', action='store_true', help='Store the alignment matrix.')
    parser.add_argument('--output_alignment', nargs='?', default='output/alignment_matrix/arenas/arenas950-1', help='Output path for alignment matrix.')


    # Node Embedding
    parser.add_argument('--embmethod', nargs='?', default='netMF', help='Node embedding method.')
    # netMF parameters
    parser.add_argument("--rank", default=256, type=int,
                        help='Number of eigenpairs used to approximate normalized graph Laplacian.')
    parser.add_argument("--dim", default=128, type=int, help='Dimension of embedding.')
    parser.add_argument("--window", default=10, type=int, help='Context window size.')
    parser.add_argument("--negative", default=1.0, type=float, help='Number of negative samples.')
    
    parser.add_argument('--store_emb', action='store_true', help='Store the node embedding.')
    parser.add_argument('--embeddingA', nargs='?', default='emb/netMF/arenas/arenas950-1.graph1.npy', help='Node embedding path for the first graph.')
    parser.add_argument('--embeddingB', nargs='?', default='emb/netMF/arenas/arenas950-1.graph2.npy', help='Node embedding path for the second graph.')

    # Embedding Space Alignment
    # convex initialization parameters
    parser.add_argument('--niter_init', type=int, default=50, help='Number of iterations.')
    parser.add_argument('--reg_init', type=float, default=1.0, help='Regularization parameter.')
    # WP optimization parameters
    parser.add_argument('--nepoch', type=int, default=300, help='Number of epochs.')
    parser.add_argument('--niter_align', type=int, default=50, help='Iterations per epoch.')
    parser.add_argument('--reg_align', type=float, default=0.05, help='Regularization parameter.')
    parser.add_argument('--bsz', type=int, default=10, help='Batch size.')
    
    parser.add_argument('--lr', type=float, default=1, help='Learning rate.')


    # Matching Nodes
    parser.add_argument('--embsim', nargs='?', default='euclidean', help='Metric for comparing embeddings.')
    parser.add_argument('--alignmethod', nargs='?', default='greedy', help='Method to align embeddings.')
    parser.add_argument('--numtop', type=int, default=10,
                      help='Number of top similarities to compute with kd-tree.  If None, computes all pairwise similarities.')

    return parser.parse_args()

def align_embeddings(embed1, embed2, adj1 = None, adj2 = None, struc_embed = None, struc_embed2 = None):
    # Step 2: Align Embedding Spaces
    corr = None
    if struc_embed is not None and struc_embed2 is not None:
        if args.embsim == "cosine":
            corr = sklearn.metrics.pairwise.cosine_similarity(embed1, embed2)
        else:
            corr = sklearn.metrics.pairwise.euclidean_distances(embed1, embed2)
            corr = np.exp(-corr)

        # Take only top correspondences
        matches = np.zeros(corr.shape)
        matches[np.arange(corr.shape[0]), np.argmax(corr, axis = 1)] = 1
        corr = matches

    # Convex Initialization
    if adj1 is not None and adj2 is not None:
        if not sps.issparse(adj1): adj1 = sps.csr_matrix(adj1)
        if not sps.issparse(adj2): adj2 = sps.csr_matrix(adj2)
        init_sim, corr_mat = unsup_align.convex_init_sparse(embed1, embed2, K_X = adj1, K_Y = adj2, apply_sqrt = False, niter = args.niter_init, reg = args.reg_init, P = corr)
    else:
        init_sim, corr_mat = unsup_align.convex_init(embed1, embed2, apply_sqrt = False, niter = args.niter_init, reg = args.reg_init, P = corr)


    # Stochastic Alternating Optimization
    dim_align_matrix, corr_mat = unsup_align.align(embed1, embed2, init_sim, lr = args.lr, bsz = args.bsz, nepoch = args.nepoch, niter = args.niter_align, reg = args.reg_align)

    # Step 3: Match Nodes with Similar Embeddings
    # Align embedding spaces
    aligned_embed1 = embed1.dot(dim_align_matrix)
    # Greedily match nodes
    if args.alignmethod == 'greedy':  # greedily align each embedding to most similar neighbor
        # KD tree with only top similarities computed
        if args.numtop is not None:
            alignment_matrix = kd_align(aligned_embed1, embed2, distance_metric=args.embsim, num_top=args.numtop)
        # All pairwise distance computation
        else:
            if args.embsim == "cosine":
                alignment_matrix = sklearn.metrics.pairwise.cosine_similarity(aligned_embed1, embed2)
            else:
                alignment_matrix = sklearn.metrics.pairwise.euclidean_distances(aligned_embed1, embed2)
                alignment_matrix = np.exp(-alignment_matrix)
    return alignment_matrix


def get_counterpart(alignment_matrix, true_alignments, part_nodes):
    n_nodes = alignment_matrix.shape[0]
    
    correct_0_nodes = []
    correct_1_nodes = []
    incorrect_0_nodes = []
    incorrect_1_nodes = []
    counterpart_dict = {}

    if not sps.issparse(alignment_matrix):
        sorted_indices = np.argsort(alignment_matrix)

    for node_index in range(n_nodes):
        target_alignment = node_index #default: assume identity mapping, and the node should be aligned to itself
        if true_alignments is not None: #if we have true alignments (which we require), use those for each node
            target_alignment = int(true_alignments[node_index])
        if sps.issparse(alignment_matrix):
            row, possible_alignments, possible_values = sps.find(alignment_matrix[node_index])
            node_sorted_indices = possible_alignments[possible_values.argsort()]
        else:
            node_sorted_indices = sorted_indices[node_index]
        if (target_alignment in part_nodes) and (node_sorted_indices[-1:][0] in part_nodes):
            correct_0_nodes.append(node_index)
        elif (target_alignment in part_nodes) and (node_sorted_indices[-1:][0] not in part_nodes):
            incorrect_0_nodes.append(node_index)
        elif (target_alignment not in part_nodes) and (node_sorted_indices[-1:][0] in part_nodes):
            incorrect_1_nodes.append(node_index)
        elif (target_alignment not in part_nodes) and (node_sorted_indices[-1:][0] not in part_nodes):
            correct_1_nodes.append(node_index)
        counterpart = node_sorted_indices[-1]
        counterpart_dict[node_index] = counterpart

    #part_nodes er de rigtige nodes i queryen - y_true
    #correct 0 er dem som vi har gættet korrekt er inde i query - TP
    #correct 1 er dem som vi har gættet korrekt er udenfor query - TN
    # incorrect 0 er dem som vi har gættet forkert er inde i query - FN
    # incorrect 1 er dem som vi har gættet forkert er udenfor query - FP

    print(correct_0_nodes, correct_1_nodes)
    print("")
    print(incorrect_0_nodes, incorrect_1_nodes)
    print(len(correct_0_nodes)+len(incorrect_0_nodes))
    balanced_accuracy = (len(correct_0_nodes)/len(part_nodes) + len(correct_1_nodes)/(n_nodes-len(part_nodes)))/2.0
    accuracy = len(correct_0_nodes)/len(part_nodes)
    if (len(correct_0_nodes)+len(incorrect_0_nodes) == 0):
        precision = 0
    else:
        precision = len(correct_0_nodes) / (len(correct_0_nodes)+len(incorrect_1_nodes))
    if (len(correct_0_nodes)+len(incorrect_0_nodes)==0):
        recall = 0
    else:
        recall = len(correct_0_nodes) / (len(correct_0_nodes)+len(incorrect_0_nodes))

    if (precision+recall == 0):
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    print(precision, recall, f1)
    return balanced_accuracy, accuracy, f1


def kd_align(emb1, emb2, normalize=False, distance_metric="euclidean", num_top=10):
    kd_tree = KDTree(emb2, metric=distance_metric)

    row = np.array([])
    col = np.array([])
    data = np.array([])

    dist, ind = kd_tree.query(emb1, k=num_top)
    print("queried alignments")
    row = np.array([])
    for i in range(emb1.shape[0]):
        row = np.concatenate((row, np.ones(num_top) * i))
    col = ind.flatten()
    data = np.exp(-dist).flatten()
    sparse_align_matrix = coo_matrix((data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))
    return sparse_align_matrix.tocsr()


def main(args):
    import os
    graph_names = ['football', 'highschool', 'malaria', 'powerlaw_200_50_50', 'renyi_200_50', 'barabasi_200_50']
    for graphname in graph_names:
        res_dict={}
        res_dict[graphname] = {}
        balanced_accuracies = []
        conductances = []
        accuracies = []
        f1s = []
        for nofolder in range(2, 6):
            subsizes = [0.1, 0.2, 0.3]
            pers = [i/10.0 for i in range(1,11)]
            for subsize in subsizes:
                res_dict[graphname][int(subsize*100)] = {}
                for per in pers: 
                    edge_list_G1 = f'./data/{graphname}/{int(subsize*100)}/{nofolder}/{graphname}_{int(subsize*100)}_{int(per*100)}.txt'
                    print(f'************* Reading file {edge_list_G1}')
                    partfile = f'./data/{graphname}/{int(subsize*100)}/{nofolder}/{graphname}_{int(subsize*100)}_nodes.txt'
                    part_nodes=np.loadtxt(partfile)



                    adjA = edgelist_to_adjmatrix(edge_list_G1)
                    G_full=nx.from_numpy_array(adjA)
                    
                    print(G_full.number_of_nodes(), G_full.number_of_edges())
                                        
                    G_part=nx.induced_subgraph(G_full,part_nodes)
                    condac = nx.conductance(G_full, part_nodes)

                    adjB_cur=nx.to_numpy_array(G_part)
                    adjB=np.zeros(np.shape(adjA))

                    for row in range(np.shape(adjB_cur)[0]):
                        for col in range(np.shape(adjB_cur)[0]):
                            adjB[row,col]=adjB_cur[row,col]


                    # step1: obtain normalized proximity-preserving node embeddings
                    if (args.embmethod == "netMF"):
                        flag = True
                        while flag:
                            try:
                                emb_matrixA = embedding.netmf(adjA, dim = min(128, np.shape(adjB)[0]-1), window=args.window, b=args.negative, normalize=True)
                                flag = False
                            except:
                                print("An exception occurred")
                        flag = True
                        while flag:
                            try:
                                emb_matrixB = embedding.netmf(adjB, dim = min(128, np.shape(adjB)[0]-1), window=args.window, b=args.negative, normalize=True)
                                flag = False
                            except:
                                print("An exception occurred")
                        #emb_matrixA = embedding.netmf(adjA, dim = min(100, np.shape(adjB)[0]-1), window=args.window, b=args.negative, normalize=True)
                        #emb_matrixB = embedding.netmf(adjB, dim = min(100, np.shape(adjB)[0]-1), window=args.window, b=args.negative, normalize=True)
                        after_emb = time.time()
                        if (args.store_emb):
                            np.save(args.embeddingA, emb_matrixA, allow_pickle=False)
                            np.save(args.embeddingB, emb_matrixB, allow_pickle=False)

                        # step2 and 3: align embedding spaces and match nodes with similar embeddings
                        alignment_matrix = align_embeddings(emb_matrixA, emb_matrixB, adj1=csr_matrix(adjA), adj2=csr_matrix(adjB), struc_embed=None, struc_embed2=None)


                        # evaluation
                        true_align = None
                        balanced_accuracy, accuracy, f1 = get_counterpart(alignment_matrix, true_align, part_nodes)
                        print("Accuracy of CONE-align: %f" % accuracy)
                        print("Balanced Accuracy of CONE-align: %f" % balanced_accuracy)
                        print("f1 of CONE-align: %f" % f1)

                        balanced_accuracies.append(balanced_accuracy)
                        accuracies.append(accuracy)
                        f1s.append(f1)
                        conductances.append(condac)

                        res_dict[graphname][(int(subsize*100))][condac] = [accuracy, balanced_accuracy, f1]
                
                
                rel_path = f'experiments/{graphname}/{subsize}'
                Path(rel_path).mkdir(parents=True, exist_ok=True)
                script_dir = os.path.dirname(__file__)
                abs_file_path = os.path.join(script_dir, rel_path)
                if not os.path.exists(abs_file_path):
                    os.makedirs(abs_file_path)
                f = open(f'{abs_file_path}/balanced_accuracies.txt', 'w')
                f.write(str(balanced_accuracies))
                f = open(f'{abs_file_path}/accuracies.txt', 'w')
                f.write(str(accuracies))
                f = open(f'{abs_file_path}/f1s.txt', 'w')
                f.write(str(f1s))
                f = open(f'{abs_file_path}/conductances.txt', 'w')
                f.write(str(conductances))
                    
            
            if True:
                print(res_dict)
                name_to_save = graphname+"_"+str(nofolder)
                directo = './pkl_results_CONE/'+graphname+'/'+name_to_save+'.pkl'
                import os
                if not os.path.exists('./pkl_results_CONE/'+graphname):
                        os.makedirs('./pkl_results_CONE/'+graphname)
                with open(directo, 'wb') as f:
                    pickle.dump(res_dict, f)
            
       
       
 





if __name__ == "__main__":
    args = parse_args()
    main(args)
