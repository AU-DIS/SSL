import sys
sys.path.insert(0, '..')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import networkx as nx
import torch
from problem.spectral_subgraph_localization import edgelist_to_adjmatrix
from optimization.prox.prox import ProxSphere, ProxL21ForSymmetricCenteredMatrix
from problem.spectral_subgraph_localization import SubgraphIsomorphismSolver, VotingSubgraphIsomorpishmSolver, VotingSubgraphIsomorpishmSolver, Solution_algo
import pickle
import sys
from copy import deepcopy

from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support

def balanced_acc(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)

def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)

def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label=0)

def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=0)

def prec_recall_fscore(y_true, y_pred):
    prec, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred)
    return prec, recall, fscore

def accur(y_true, y_pred):
    counter = 0
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == 0:
            counter += 1
            if y_true[i] == y_pred[i]:
                correct += 1
    return 1.0*correct/counter

def prune_graph(G, part_nodes):
    lowest_degree = np.inf
    sub_G = nx.subgraph(G, part_nodes)
    for degree_view in nx.degree(sub_G):
        lowest_degree = min(lowest_degree,degree_view[1])
    print("removing edges:", list(n for n in G.nodes if nx.degree(G,n)<lowest_degree))
    G.remove_nodes_from(list(n for n in G.nodes if nx.degree(G,n)<lowest_degree))

def run_opt(edgefile,part_nodes, mu=1, standard_voting_thresholds=[], neighborhood_thresholds=[]):
    print(f'Reading from {edgefile}')
    A1=edgelist_to_adjmatrix(edgefile)
    G=nx.from_numpy_matrix(A1)
    A = torch.tensor(nx.to_numpy_matrix(G))

    n1 = len(part_nodes)
    n = len(G.nodes)
    print(f"|Vq| = {n1}")
    print(f"|V| = {n}")
    print(f"|Vq|/|V| = {n1/n}")
    
    condac = nx.conductance(G, part_nodes)
    print(f"Conductance equals to {condac}")

    # prune_graph(G, part_nodes) # TODO gør noget med G, lav A ud fra G? Vær opmærksom på om originale indices stadig passer...

    color_map=[]
    for node in G:
        if node in part_nodes:
            color_map.append('blue')
        else: 
            color_map.append('green') 
    
    D = torch.diag(A.sum(dim=1))
    L = D - A

    A_sub = A[0:n1, 0:n1]
    A_sub = A[:,part_nodes]
    A_sub=A_sub[part_nodes,:]

    D_sub = torch.diag(A_sub.sum(dim=1))
    L_sub = D_sub - A_sub
    ref_spectrum = torch.linalg.eigvalsh(L_sub)

    v_val = float(np.max(ref_spectrum.numpy()))
    c = np.sqrt(n-len(part_nodes)) * v_val
    v_gt = v_val * np.ones(n)

    for i in range(n):
        if(i in part_nodes):
            v_gt[i] = 0.0

    problem_params = {'mu_spectral': 1,
                      'mu_l21': 0,
                      'mu_MS': 0,
                      'mu_split': 0,
                      'mu_trace': 0.0,
                      'trace_val': 0,
                      'weighted_flag': False
                      }

    solver_params={}


    solver_params = {'lr': 0.02, #learning rate
                 'a_tol': -1, # not used because its negative
                 'r_tol': -1e-5/c**2, # not used because its negative
                 'v_prox': ProxSphere(radius=c), #projection on a sphere
                 #'v_prox': ProxId(),
                 'E_prox': ProxL21ForSymmetricCenteredMatrix(solver="cvx"), #not used
                 # 'E_prox': ProxL1ForSymmCentdMatrixAndInequality(solver="cvx", L=L,
                 #                                                 trace_upper_bound=
                 #                                                 1.1 * torch.trace(L)),}
                 'train_v': True, #TODO....
                 'train_E': False, #TODO...
                 'threshold_algo': '1dkmeans', # 'spectral', '1dkmeans', 'smallest'
                 }


    subgraph_isomorphism_solver = SubgraphIsomorphismSolver(A, ref_spectrum, problem_params, solver_params)
    v, E = \
        subgraph_isomorphism_solver.solve(max_outer_iters=3,max_inner_iters=500, show_iter=10000, verbose=False)
    v_binary, E_binary = subgraph_isomorphism_solver.threshold(v_np=v.detach().numpy())
    
    gt_inidicator = v_gt
    gt_inidicator[gt_inidicator>0]=1 
    original_balanced = balanced_acc(v_gt, v_binary.clone().detach().numpy())

    problem_params = {'mu_spectral': 1,
                      'mu_l21': 0,
                      'mu_MS': mu,# / (c ** 2), #The μ regularizer eq. 11
                      'mu_split': 0,
                      'mu_trace': 0.0,
                      'trace_val': 0,
                      'weighted_flag': False
                      }
    subgraph_isomorphism_solver.set_problem_params(problem_params)

    v_binary, E_binary = subgraph_isomorphism_solver.threshold(v_np=v.detach().numpy())
    idx_smallest=np.argsort(v.detach().numpy())[:n1]
    v_smallest=np.ones((n,1))
    for i in range(n):
        if i in idx_smallest:
            v_smallest[i]=0

    original_accuracy = accur(v_gt, v_binary.clone().detach().numpy())
    original_balanced = balanced_acc(v_gt, v_binary.clone().detach().numpy())
    og_precision, og_recall, og_fscore = prec_recall_fscore(v_gt, v_binary.clone().detach().numpy())

    experiments_to_make = 20
    random_solver = VotingSubgraphIsomorpishmSolver(A, ref_spectrum, problem_params, solver_params, v_gt, A_sub, experiments_to_make) # Faked original balanced accuracy, can probably delete anyway
    # v_randomized, _ = random_solver.solve(max_outer_iters=3,max_inner_iters=500, show_iter=10000, verbose=False)
    votes = random_solver.solve(max_outer_iters=3,max_inner_iters=500, show_iter=10000, verbose=False)


    standard_voting_results = []
    for threshold in standard_voting_thresholds:
        v, _ = VotingSubgraphIsomorpishmSolver.find_solution(A, votes, experiments_to_make, Solution_algo.THRESHOLD, threshold = threshold)
        v_accuracy = accur(v_gt, v.clone().detach().numpy())
        v_balanced_accuracy = balanced_acc(v_gt, v.clone().detach().numpy())
        v_recall = recall(v_gt, v.clone().detach().numpy())
        v_precision = precision(v_gt, v.clone().detach().numpy())
        v_fscore = f1(v_gt, v.clone().detach().numpy())

        standard_voting_results.append({
                "threshold": threshold,
                "acc": v_accuracy,
                "balanced_acc": v_balanced_accuracy,
                "recall": v_recall,
                "precision": v_precision,
                "f1": v_fscore,
            })

    neighborhood_results = []
    for threshold in neighborhood_thresholds:
        v, _ = VotingSubgraphIsomorpishmSolver.find_solution(A, votes, experiments_to_make, Solution_algo.DIJKSTRA, threshold_percentage = threshold)
        v_accuracy = accur(v_gt, v.clone().detach().numpy())
        v_balanced_accuracy = balanced_acc(v_gt, v.clone().detach().numpy())
        v_recall = recall(v_gt, v.clone().detach().numpy())
        v_precision = precision(v_gt, v.clone().detach().numpy())
        v_fscore = f1(v_gt, v.clone().detach().numpy())

        neighborhood_results.append({
                "threshold": threshold,
                "acc": v_accuracy,
                "balanced_acc": v_balanced_accuracy,
                "recall": v_recall,
                "precision": v_precision,
                "f1": v_fscore,
            })

    # Returning original accuracy
    return standard_voting_results, neighborhood_results, condac

def count_nodes(v_binary):
    return len(v_binary) - np.count_nonzero(v_binary)

def find_best_mu(edgefile,part_nodes):
    print(f'Reading from {edgefile}')
    A1=edgelist_to_adjmatrix(edgefile)
    G=nx.from_numpy_matrix(A1)
    A = torch.tensor(nx.to_numpy_matrix(G))

    n1 = len(part_nodes)
    n = len(G.nodes)
    print(f"|Vq| = {n1}")
    print(f"|V| = {n}")
    print(f"|Vq|/|V| = {n1/n}")
    
    condac = nx.conductance(G, part_nodes)
    print(f"Conductance equals to {condac}")
    print(f'query nodes {part_nodes}')
    color_map=[]
    for node in G:
        if node in part_nodes:
            color_map.append('blue')
        else: 
            color_map.append('green') 
    
    D = torch.diag(A.sum(dim=1))
    L = D - A

    A_sub = A[0:n1, 0:n1]
    A_sub = A[:,part_nodes]
    A_sub=A_sub[part_nodes,:]
    #print(A_sub)

    D_sub = torch.diag(A_sub.sum(dim=1))
    L_sub = D_sub - A_sub
    ref_spectrum = torch.linalg.eigvalsh(L_sub)

    v_val = float(np.max(ref_spectrum.numpy()))
    c = np.sqrt(n-len(part_nodes)) * v_val
    v_gt = v_val * np.ones(n)

    for i in range(n):
        if(i in part_nodes):
            v_gt[i] = 0.0

    problem_params = {'mu_spectral': 1,
                      'mu_l21': 0,
                      'mu_MS': 0,
                      'mu_split': 0,
                      'mu_trace': 0.0,
                      'trace_val': 0,
                      'weighted_flag': False
                      }

    solver_params = {'lr': 0.02, #learning rate
                 'a_tol': -1, # not used because its negative
                 'r_tol': -1e-5/c**2, # not used because its negative
                 'v_prox': ProxSphere(radius=c), #projection on a sphere
                 #'v_prox': ProxId(),
                 'E_prox': ProxL21ForSymmetricCenteredMatrix(solver="cvx"), #not used
                 # 'E_prox': ProxL1ForSymmCentdMatrixAndInequality(solver="cvx", L=L,
                 #                                                 trace_upper_bound=
                 #                                                 1.1 * torch.trace(L)),}
                 'train_v': True, #TODO....
                 'train_E': False, #TODO...
                 'threshold_algo': '1dkmeans', # 'spectral', '1dkmeans', 'smallest'
                 }


    subgraph_isomorphism_solver = SubgraphIsomorphismSolver(A, ref_spectrum, problem_params, solver_params)
    pp = [i/10.0 for i in range(11)]
    
    best_mu = 0
    best_ba = 0
    for  mu_MS in pp:
        print(f"iter = {iter}, mu_MS = {mu_MS}")
        problem_params = {'mu_spectral': 1,
                          'mu_l21': 0,
                          'mu_MS': mu_MS,# / (c ** 2), #The μ regularizer eq. 11
                          'mu_split': 0,
                          'mu_trace': 0.0,
                          'trace_val': 0,
                          'weighted_flag': False
                          }
        subgraph_isomorphism_solver.set_problem_params(problem_params)
        v, E = \
            subgraph_isomorphism_solver.solve(max_outer_iters=3,max_inner_iters=50, show_iter=10000, verbose=False)
        v_binary, E_binary = subgraph_isomorphism_solver.threshold(v_np=v.detach().numpy())
        for i in range(len(v_gt)):
            if v_gt[i] > 0: v_gt[i] = 1
            else: v_gt[i] = 0
        ba_current = balanced_acc(v_gt, v_binary.clone().detach())
        
        
        if ba_current>best_ba:
            best_ba = ba_current
            best_mu = mu_MS
            counter = 0
        else:
            counter += 1
        print("Searching for the best balance accuracy...")
        print(f"Current Balanced Accuracy for {mu_MS} is {ba_current}")
        print(f"Best Balanced Accuracy is for {best_mu}, equal to {best_ba}")
        if((counter>10) or (best_ba==1.0)): break
    return best_mu

if __name__ == '__main__':
    #graph_names = ['ant', 'football', 'highschool', 'malaria', 'powerlaw_200_50_50', 'renyi_200_50', 'barabasi_200_50']
    graph_names = ['football']
    standard_voting_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    neighborhood_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    conductances = []

    initial_dict = {threshold: [] for threshold in standard_voting_thresholds}

    # Create dictionaries for standard voting
    standard_voting_balanced_accuracies = deepcopy(initial_dict)
    standard_voting_accuracies = deepcopy(initial_dict)
    standard_voting_recalls = deepcopy(initial_dict)
    standard_voting_precisions = deepcopy(initial_dict)
    standard_voting_f1s = deepcopy(initial_dict)

    # Create dictionaries for neighborhood
    neighborhood_balanced_accuracies = deepcopy(initial_dict)
    neighborhood_accuracies = deepcopy(initial_dict)
    neighborhood_recalls = deepcopy(initial_dict)   
    neighborhood_precisions = deepcopy(initial_dict)
    neighborhood_f1s = deepcopy(initial_dict)

    graphs = []
    use_global_mu = True
    for graph_name in graph_names:
        res_dict={}
        res_dict[graph_name] = {}
        best_mu = {}
        best_m_par = 0
        counter_m_par = 0
        folderAmount = 2
        for folder_no in range(0,folderAmount):
            if use_global_mu and folder_no==0: continue
            perc = [0.1, 0.2, 0.3]
            clcr = [i/10.0 for i in range(1, 11)]
            for per in perc:
                if folder_no == 0: 
                    best_mu[per] = {}
                res_dict[graph_name][int(per*100)] = {}
                for lr in clcr:   
                    if folder_no == 0: 
                        best_mu[per][lr] = 0
                    ext = str(int(per*100))
                    part_file = './data/'+graph_name+'/'+str(int(per*100))+'/'+str(folder_no)+'/'+graph_name+'_'+ext+'_nodes.txt'
                    print(f'Reading subgraph from {part_file}')
                    query_nodes=np.loadtxt(part_file)
                    print(query_nodes)
                    #print(part_nodes)
                    ext = str(int(per*100))+"_"+str(int(lr*100))
                    edgefile = './data/'+graph_name+'/'+str(int(per*100))+'/'+str(folder_no)+'/'+graph_name+'_'+ext+'.txt'
                    if folder_no == 0:
                        print('**** Looking for the best m ****')
                        best_mu[per][lr] = find_best_mu(edgefile, query_nodes)
                        best_m_par += best_mu[per][lr]
                        counter_m_par += 1
                    else:
                        if use_global_mu:
                            # acc, bal_acc, condac, recall_s, precision_s, f1_s =run_opt(edgefile,query_nodes, 0.2, standard_voting_thresholds, neighborhood_thresholds)
                            standard_voting_results, neighborhood_results, condac = run_opt(edgefile,query_nodes, 0.2, standard_voting_thresholds, neighborhood_thresholds)
                            conductances.append(condac)
                            # res_dict[graph_name][(int(per*100))][condac] = [acc, bal_acc, 0.2]
                            for result in standard_voting_results:
                                threshold = result["threshold"]
                                standard_voting_balanced_accuracies[threshold].append(result["balanced_acc"])   
                                standard_voting_accuracies[threshold].append(result["acc"])
                                standard_voting_recalls[threshold].append(result["recall"])
                                standard_voting_precisions[threshold].append(result["precision"])
                                standard_voting_f1s[threshold].append(result["f1"])
                            for result in neighborhood_results:
                                threshold = result["threshold"]
                                neighborhood_balanced_accuracies[threshold].append(result["balanced_acc"])   
                                neighborhood_accuracies[threshold].append(result["acc"])
                                neighborhood_recalls[threshold].append(result["recall"])
                                neighborhood_precisions[threshold].append(result["precision"])
                                neighborhood_f1s[threshold].append(result["f1"])
                        else:
                            standard_voting_results, neighborhood_results, condac =run_opt(edgefile,query_nodes, best_mu[per][lr], standard_voting_thresholds, neighborhood_thresholds)
                            conductances.append(condac)
                            # res_dict[graph_name][(int(per*100))][condac] = [acc, bal_acc, 0.2]
                            for result in standard_voting_results:
                                threshold = result["threshold"]
                                standard_voting_balanced_accuracies[threshold].append(result["balanced_acc"])   
                                standard_voting_accuracies[threshold].append(result["acc"])
                                standard_voting_recalls[threshold].append(result["recall"])
                                standard_voting_precisions[threshold].append(result["precision"])
                                standard_voting_f1s[threshold].append(result["f1"])
                            for result in neighborhood_results:
                                threshold = result["threshold"]
                                neighborhood_balanced_accuracies[threshold].append(result["balanced_acc"])   
                                neighborhood_accuracies[threshold].append(result["acc"])
                                neighborhood_recalls[threshold].append(result["recall"])
                                neighborhood_precisions[threshold].append(result["precision"])
                                neighborhood_f1s[threshold].append(result["f1"])
            print(res_dict)
            f = open('condutance.txt', 'a+')
            f.write(str(conductances))

            # Write results for standard voting 
            for threshold, values in standard_voting_balanced_accuracies.items():
                f = open(f'balanced_accuracy_{threshold}.txt', 'w+')
                f.write(str(values))

            for threshold, values in standard_voting_accuracies.items():
                f = open(f'accuracy_{threshold}.txt', 'w+')
                f.write(str(values))

            for threshold, values in standard_voting_recalls.items():
                f = open(f'recall_{threshold}.txt', 'w+')
                f.write(str(values))

            for threshold, values in standard_voting_precisions.items():
                f = open(f'precision_{threshold}.txt', 'w+')
                f.write(str(values))

            for threshold, values in standard_voting_f1s.items():
                f = open(f'f1_{threshold}.txt', 'w+')
                f.write(str(values))

            # Write results for neighborhood
            for threshold, values in neighborhood_balanced_accuracies.items():
                f = open(f'n_balanced_accuracy_{threshold}.txt', 'w+')
                f.write(str(values))

            for threshold, values in neighborhood_accuracies.items():
                f = open(f'n_accuracy_{threshold}.txt', 'w+')
                f.write(str(values))

            for threshold, values in neighborhood_recalls.items():
                f = open(f'n_recall_{threshold}.txt', 'w+')
                f.write(str(values))

            for threshold, values in neighborhood_precisions.items():
                f = open(f'n_precision_{threshold}.txt', 'w+')
                f.write(str(values))

            for threshold, values in neighborhood_f1s.items():
                f = open(f'n_f1_{threshold}.txt', 'w+')
                f.write(str(values))

            data_file = open(f'{graph_name}.txt', 'w')
            data_file.write(str(res_dict))

            if(True):
                name_to_save = graph_name+"_"+str(folder_no)
                directo = './pkl_results/'+graph_name+'/'+name_to_save+'.pkl'
                import os
                if not os.path.exists('./pkl_results/'+graph_name):
                        os.makedirs('./pkl_results/'+graph_name)
                with open(directo, 'wb') as f:
                    pickle.dump(res_dict, f)
            #sys.exit(


