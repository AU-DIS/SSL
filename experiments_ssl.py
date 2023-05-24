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
from pathlib import Path
import math

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

def graph_edit_distance(y_true, y_pred):
    return nx.optimize_graph_edit_distance(y_true, y_pred)

def use_graph_edit_distance_generator(generator_object, description=None):
    num_iterations = 0
    if description is None:
        description = ""

    distance = None
    for distance in generator_object:
        num_iterations += 1
        print(distance, description)
        if num_iterations == 3:
            break
    
    return distance

def solution_graph(G, solution_vector):
    _G = G.copy()
    solution_indices = [i for i, res in enumerate(solution_vector) if res == 0]
    S = _G.subgraph(solution_indices)
    return S

def spectrum_from_graph(G):
    A = torch.tensor(nx.to_numpy_matrix(G))
    D = torch.diag(A.sum(dim=1))
    L = D - A
    return torch.linalg.eigvalsh(L)

# Calculates the sum of the absolute entry-wise difference between two torch tensors.
# If |X| < |Y|, the last eigenvalue of G2 is copied, untill the length of both lists are of same length.
def spectrum_abs_diff(X, Y):
    if len(X) > len(Y):
        return 999999

    eigenvalues_to_compare = min(len(X), len(Y)) # If 
    Y = Y[:eigenvalues_to_compare]
    X = X[:eigenvalues_to_compare]
    return torch.sum(torch.abs(torch.sub(X, Y))).item()

def greedy_remove_node_by_spectrum(G, solution_vector, ref_spectrum):
    solution_indices = [i for i, res in enumerate(solution_vector) if res == 0]
    return greedy_remove_node_by_spectrum_aux(G, solution_indices, ref_spectrum)

def greedy_remove_node_by_spectrum_aux(G, solution_indices, ref_spectrum):
    if len(solution_indices) == len(ref_spectrum):
        return solution_indices

    smallest_spectrum_diff = math.inf
    best_idx_to_remove = None
    for idx in solution_indices:
        new_solution_indices = solution_indices.copy()
        new_solution_indices.remove(idx)
        new_solution = G.subgraph(new_solution_indices)
        new_spectrum = spectrum_from_graph(new_solution)
        new_spectrum_diff = spectrum_abs_diff(ref_spectrum, new_spectrum)
        if new_spectrum_diff < smallest_spectrum_diff:
            smallest_spectrum_diff = new_spectrum_diff
            best_idx_to_remove = idx

    solution_indices.remove(best_idx_to_remove)
    return greedy_remove_node_by_spectrum_aux(G, solution_indices, ref_spectrum)

def greedy_add_node_by_spectrum_v2(G, solution_vector, ref_spectrum):
    solution_indices = [i for i, res in enumerate(solution_vector) if res == 0]
    remaining_indices = [i for i, res in enumerate(solution_vector) if res == 1]
    return greedy_add_node_by_spectrum_v2_aux(G, solution_indices, remaining_indices, ref_spectrum)

def greedy_add_node_by_spectrum_v2_aux(G, solution_indices, remaining_indices, ref_spectrum):
    if len(solution_indices) == len(ref_spectrum):
        return solution_indices

    smallest_spectrum_diff = math.inf
    best_idx_to_add = None
    for idx in remaining_indices:
        new_solution_indices = solution_indices.copy()
        new_solution_indices.append(idx)
        new_solution = G.subgraph(new_solution_indices)
        new_spectrum = spectrum_from_graph(new_solution)
        new_spectrum_diff = spectrum_abs_diff(ref_spectrum, new_spectrum)
        if new_spectrum_diff < smallest_spectrum_diff:
            smallest_spectrum_diff = new_spectrum_diff
            best_idx_to_add = idx

    solution_indices.append(best_idx_to_add)
    remaining_indices.remove(best_idx_to_add)
    return greedy_add_node_by_spectrum_v2_aux(G, solution_indices, remaining_indices, ref_spectrum)

# TODO clean up kode, så vi ikke har aux metoder, men at de pågældende metoder bare bliver kaldt herfra
# TODO overvej at restricte greedy add nodes, så den kun overvejer nodes, der faktisk er blevet stemt på
# TODO ændr hvordan vi sammenligner spektrum når |S| < |Q|, ifht hvad end Petros har skrevet
def enforce_cardinality_constraint_by_spectrum(G, solution_vector, ref_spectrum):
    solution_indices = [i for i, res in enumerate(solution_vector) if res == 0]

    nodes_solution = len(solution_indices)
    nodes_query = len(ref_spectrum)

    final_solution_indices = None
    if nodes_solution == nodes_query:
        final_solution_indices = solution_indices
    elif nodes_solution < nodes_query:
        remaining_indices = [i for i, res in enumerate(solution_vector) if res == 1]
        final_solution_indices = greedy_add_node_by_spectrum_v2_aux(G, solution_indices, remaining_indices, ref_spectrum)
    else:
        final_solution_indices = greedy_remove_node_by_spectrum_aux(G, solution_indices, ref_spectrum)

    final_solution_vector = [1] * len(solution_vector)
    for idx in final_solution_indices:
        final_solution_vector[idx] = 0
    return final_solution_vector

def test_greedy_remove(G, A, part_nodes, ref_spectrum):
    n, _ = A.shape

    part_nodes_altered = [node for node in part_nodes]

    nodes_to_add = [42, 69, 80, 100]
    # print("Nodes added to query:", nodes_to_add)
    part_nodes_altered.extend(nodes_to_add)
    part_nodes_altered = np.array(part_nodes_altered)

    n2 = len(part_nodes_altered)

    A_sub = A[0:n2, 0:n2]
    A_sub = A[:,part_nodes_altered]
    A_sub = A_sub[part_nodes_altered,:]

    Q_altered_solution_vector = [0 if idx in part_nodes_altered else 1 for idx in range(n)]

    enforce_cardinality_constraint_by_spectrum(G, Q_altered_solution_vector, ref_spectrum)

def test_greedy_add(G, A, part_nodes, ref_spectrum):
    n, _ = A.shape

    part_nodes_altered = [node for node in part_nodes[:-4]]

    n2 = len(part_nodes_altered)

    A_sub = A[0:n2, 0:n2]
    A_sub = A[:,part_nodes_altered]
    A_sub = A_sub[part_nodes_altered,:]
    Q_altered_solution_vector = [0 if idx in part_nodes_altered else 1 for idx in range(n)]

    enforce_cardinality_constraint_by_spectrum(G, Q_altered_solution_vector, ref_spectrum)


def test_spectrum_abs_diff():
    ref = torch.tensor([-1.874207407808E-15,	2.682128349763E-01,	9.692844069884E-01,	1.000000000000E+00,	1.888921131972E+00,	3.000000000000E+00,	3.239240861833E+00,	4.528519547368E+00,	4.584230646880E+00,	6.067163735274E+00,	6.972770944034E+00,	7.481655890675E+00])

    og = torch.tensor([-1.52483190789405E-15,	-1.05274498428316E-15,	-6.16758854339913E-16,	-3.66102675372787E-16,	1.07386426143722E-16,	6.11255181162008E-16,	1.56568879456694E-15,	1.31607933673826E-01,	2.25185406989786E-01,	6.47266783843268E-01,	2.00000000000000E+00,	2.00000000000000E+00,	2.62744224059526E+00,	3.00000000000000E+00,	3.00000000000000E+00,	3.41042274867308E+00,	3.51104938348872E+00,	4.74158760732876E+00,	5.37078044506606E+00,	5.99999999999999E+00,	6.00000000000000E+00,	6.00000000000000E+00,	6.00000000000000E+00,	6.21091198406082E+00,	7.12374546628037E+00])
    og_diff = spectrum_abs_diff(ref, og)
    expected = 34.9959398754931  # calculated in excel
    assert math.isclose(og_diff, expected, abs_tol = 1e-4)

    standard_voting = torch.tensor([3.59000232651452000E-16,	8.15644757235431000E-01,	1.06150705134328000E+00,	2.48928787911476000E+00,	2.96602525278443000E+00,	3.72664104617151000E+00,	5.00590060517529000E+00,	5.73255758501802000E+00,	6.01000779226345000E+00,	6.89624807730457000E+00,	7.25598723484175000E+00,	8.08242566678287000E+00,	8.60897879344057000E+00,	9.23614614084348000E+00,	9.86728250814203000E+00,	1.02453596095385000E+01])
    standard_voting_diff = spectrum_abs_diff(ref, standard_voting)
    expected = 10.0422329480354 # calculated in excel
    assert math.isclose(standard_voting_diff, expected, abs_tol = 1e-4)

    neighborhood = torch.tensor([-2.53356363666412000E-15,	9.11199028661349000E-01,	2.77845218364703000E+00,	3.40885154889223000E+00,	4.87057852129576000E+00,	5.19726549913658000E+00,	5.55836878028381000E+00,	6.48753597002133000E+00,	7.11976772271421000E+00,	8.06233221646075000E+00,	8.86293166343401000E+00,	9.03714889034365000E+00,	9.70556797510924000E+00])
    neighborhood_diff = spectrum_abs_diff(ref, neighborhood)
    expected = 22.2944320248908 # calculated in excel
    assert math.isclose(neighborhood_diff, expected, abs_tol = 1e-4)

    # This case testes that if there's fewer items in Y compared to X, then the last eigenvalue is added multiple times to Y
    artificial_og = torch.tensor([-1.52483190789405E-15,	-1.05274498428316E-15,	-6.16758854339913E-16,	-3.66102675372787E-16,	1.07386426143722E-16,	6.11255181162008E-16,	1.56568879456694E-15,	1.31607933673826E-01,	2.25185406989786E-01,	6.47266783843268E-01 ])
    artificial_og_diff = spectrum_abs_diff(ref, artificial_og)
    expected = 37.7014063078065  # calculated in excel
    assert math.isclose(artificial_og_diff, expected, abs_tol = 1e-4)

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
    # print("removing edges:", list(n for n in G.nodes if nx.degree(G,n)<lowest_degree))
    G.remove_nodes_from(list(n for n in G.nodes if nx.degree(G,n)<lowest_degree))

def run_opt(edgefile,part_nodes, mu=1, standard_voting_thresholds=[], neighborhood_thresholds=[], edge_removal=0.3):
    # TODO fix test?
    # test_spectrum_abs_diff()

    # print(f'Reading from {edgefile}')
    A1=edgelist_to_adjmatrix(edgefile)
    G=nx.from_numpy_matrix(A1)
    A = torch.tensor(nx.to_numpy_matrix(G))

    n1 = len(part_nodes)
    n = len(G.nodes)
    # print(f"|Vq| = {n1}")
    # print(f"|V| = {n}")
    # print(f"|Vq|/|V| = {n1/n}")
    
    condac = nx.conductance(G, part_nodes)
    # print(f"Conductance equals to {condac}")

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

    Q = nx.from_numpy_matrix(A_sub.clone().detach().numpy())

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
                      'mu_MS': mu,# / (c ** 2), #The μ regularizer eq. 11
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

    # test_greedy_remove(G, A, part_nodes, ref_spectrum)
    # test_greedy_add(G, A, part_nodes, ref_spectrum)

    subgraph_isomorphism_solver = SubgraphIsomorphismSolver(A, ref_spectrum, problem_params, solver_params)
    v, E = \
        subgraph_isomorphism_solver.solve(max_outer_iters=3,max_inner_iters=500, show_iter=10000, verbose=False)
    v_binary, E_binary = subgraph_isomorphism_solver.threshold(v_np=v.detach().numpy())
    
    gt_inidicator = v_gt
    gt_inidicator[gt_inidicator>0]=1 

    idx_smallest=np.argsort(v.detach().numpy())[:n1]
    v_smallest=np.ones((n,1))
    for i in range(n):
        if i in idx_smallest:
            v_smallest[i]=0

    original_accuracy = accur(v_gt, v_binary.clone().detach().numpy())
    original_balanced = balanced_acc(v_gt, v_binary.clone().detach().numpy())
    og_precision, og_recall, og_fscore = prec_recall_fscore(v_gt, v_binary.clone().detach().numpy())

    S = solution_graph(G, v_binary)

    # og_ged_generator = graph_edit_distance(Q, S)
    # og_ged = use_graph_edit_distance_generator(og_ged_generator, "OG")
    og_spectrum = spectrum_from_graph(S)
    og_spectrum_diff = spectrum_abs_diff(ref_spectrum, og_spectrum)
    # print("Og diff:", og_spectrum_diff)
    # print("Og balanced acc:", original_balanced)
    # print("Og f1:", og_fscore)

    nodes_in_solution = count_nodes(v_binary)
    # print("Nodes in solution:", nodes_in_solution)

    experiments_to_make = 30

    random_solver = VotingSubgraphIsomorpishmSolver(A, ref_spectrum, problem_params, solver_params, v_gt, A_sub, experiments_to_make=experiments_to_make, edge_removal=edge_removal) # Faked original balanced accuracy, can probably delete anyway
    # v_randomized, _ = random_solver.solve(max_outer_iters=3,max_inner_iters=500, show_iter=10000, verbose=False)
    votes = random_solver.solve(max_outer_iters=3,max_inner_iters=500, show_iter=10000, verbose=False)

    standard_voting_results = []
    standard_voting_results_with_cardinality_constraint = []
    for threshold in standard_voting_thresholds:
        v, _ = random_solver.find_solution(A, votes, experiments_to_make, Solution_algo.THRESHOLD, threshold = threshold)
        v_accuracy = accur(v_gt, v.clone().detach().numpy())
        v_balanced_accuracy = balanced_acc(v_gt, v.clone().detach().numpy())
        v_recall = recall(v_gt, v.clone().detach().numpy())
        v_precision = precision(v_gt, v.clone().detach().numpy())
        v_fscore = f1(v_gt, v.clone().detach().numpy())

        S = solution_graph(G, v)
        # v_ged_generator = graph_edit_distance(Q, S)
        # v_ged = use_graph_edit_distance_generator(v_ged_generator, f'Standard with threshold: {threshold}')

        # print(f'Standard voting with threshold: {threshold}')
        v_spectrum = spectrum_from_graph(S)
        v_spectrum_diff = spectrum_abs_diff(ref_spectrum, v_spectrum)

        # print("v diff:", v_spectrum_diff)
        # print("v balanced:", v_balanced_accuracy)
        # print("v fscore:", v_fscore)

        nodes_in_solution = count_nodes(v)
        # print("Nodes in solution:", nodes_in_solution)

        standard_voting_results.append({
                "threshold": threshold,
                "acc": v_accuracy,
                "balanced_acc": v_balanced_accuracy,
                "recall": v_recall,
                "precision": v_precision,
                "f1": v_fscore,
                # "graph_edit_distance": v_ged,
                "spectrum": v_spectrum.tolist(),
                "spectrum_diff": v_spectrum_diff
            })

        # Do the same with cardinality constraint enforced!
        # print("enforcing cardinality constraint")
        v = enforce_cardinality_constraint_by_spectrum(G, v, ref_spectrum)
        v_accuracy = accur(v_gt, v)
        v_balanced_accuracy = balanced_acc(v_gt, v)
        v_recall = recall(v_gt, v)
        v_precision = precision(v_gt, v)
        v_fscore = f1(v_gt, v)

        S = solution_graph(G, v)
        # v_ged_generator = graph_edit_distance(Q, S)
        # v_ged = use_graph_edit_distance_generator(v_ged_generator, f'Standard with threshold: {threshold}')

        # print(f'Standard voting with threshold: {threshold} AND cardinality constraint enforced')
        v_spectrum = spectrum_from_graph(S)
        v_spectrum_diff = spectrum_abs_diff(ref_spectrum, v_spectrum)

        # print("v diff:", v_spectrum_diff)
        # print("v balanced:", v_balanced_accuracy)
        # print("v fscore:", v_fscore)

        nodes_in_solution = count_nodes(v)
        # print("Nodes in solution:", nodes_in_solution)

        standard_voting_results_with_cardinality_constraint.append({
                "threshold": threshold,
                "acc": v_accuracy,
                "balanced_acc": v_balanced_accuracy,
                "recall": v_recall,
                "precision": v_precision,
                "f1": v_fscore,
                # "graph_edit_distance": v_ged,
                "spectrum": v_spectrum.tolist(),
                "spectrum_diff": v_spectrum_diff
            })

    neighborhood_results = []
    neighborhood_results_with_cardinality_constraint = []
    for threshold in neighborhood_thresholds:
        v, _ = random_solver.find_solution(A, votes, experiments_to_make, Solution_algo.DIJKSTRA, threshold_percentage = threshold)
        v_accuracy = accur(v_gt, v.clone().detach().numpy())
        v_balanced_accuracy = balanced_acc(v_gt, v.clone().detach().numpy())
        v_recall = recall(v_gt, v.clone().detach().numpy())
        v_precision = precision(v_gt, v.clone().detach().numpy())
        v_fscore = f1(v_gt, v.clone().detach().numpy())
        
        S = solution_graph(G, v)
        # v_ged_generator = graph_edit_distance(Q, S)
        # v_ged = use_graph_edit_distance_generator(v_ged_generator, f'Neighborhood with threshold: {threshold}')

        v_spectrum = spectrum_from_graph(S)
        # print(f'Neighborhood with threshold: {threshold}')
        v_spectrum_diff = spectrum_abs_diff(ref_spectrum, v_spectrum)

        # print("v diff:", v_spectrum_diff)
        # print("v balanced:", v_balanced_accuracy)
        # print("v fscore:", v_fscore)

        nodes_in_solution = count_nodes(v)
        # print("Nodes in solution:", nodes_in_solution)

        neighborhood_results.append({
                "threshold": threshold,
                "acc": v_accuracy,
                "balanced_acc": v_balanced_accuracy,
                "recall": v_recall,
                "precision": v_precision,
                "f1": v_fscore,
                # "graph_edit_distance": v_ged,
                "spectrum": v_spectrum.tolist(),
                "spectrum_diff": v_spectrum_diff
            })

        # Now doing the same for neighborhood with cardinality constraint!
        # print("enforcing cardinality constraint")
        v = enforce_cardinality_constraint_by_spectrum(G, v, ref_spectrum)
        v_accuracy = accur(v_gt, v)
        v_balanced_accuracy = balanced_acc(v_gt, v)
        v_recall = recall(v_gt, v)
        v_precision = precision(v_gt, v)
        v_fscore = f1(v_gt, v)
        
        S = solution_graph(G, v)
        # v_ged_generator = graph_edit_distance(Q, S)
        # v_ged = use_graph_edit_distance_generator(v_ged_generator, f'Neighborhood with threshold: {threshold}')

        v_spectrum = spectrum_from_graph(S)
        print(f'Neighborhood with threshold: {threshold} and cardinality constraint enforced')
        v_spectrum_diff = spectrum_abs_diff(ref_spectrum, v_spectrum)

        print("v diff:", v_spectrum_diff)
        print("v balanced:", v_balanced_accuracy)
        print("v fscore:", v_fscore)

        nodes_in_solution = count_nodes(v)
        print("Nodes in solution:", nodes_in_solution)

        neighborhood_results_with_cardinality_constraint.append({
                "threshold": threshold,
                "acc": v_accuracy,
                "balanced_acc": v_balanced_accuracy,
                "recall": v_recall,
                "precision": v_precision,
                "f1": v_fscore,
                # "graph_edit_distance": v_ged,
                "spectrum": v_spectrum.tolist(),
                "spectrum_diff": v_spectrum_diff
            })

    og_results = {
                "acc": original_accuracy,
                "balanced_acc": original_balanced,
                "precision": og_precision[0],
                "recall": og_recall[0],
                "f1": og_fscore[0],
                "graph_edit_distance": og_ged,
                "spectrum": og_spectrum.tolist(),
                "spectrum_diff": og_spectrum_diff
            }

    # Returning original accuracy
    return standard_voting_results, neighborhood_results, condac, og_results, ref_spectrum.tolist(), standard_voting_results_with_cardinality_constraint, neighborhood_results_with_cardinality_constraint, votes, v_gt

def count_nodes(v_binary):
    return len(v_binary) - np.count_nonzero(v_binary)

def find_best_mu(edgefile,part_nodes):
    # print(f'Reading from {edgefile}')
    A1=edgelist_to_adjmatrix(edgefile)
    G=nx.from_numpy_matrix(A1)
    A = torch.tensor(nx.to_numpy_matrix(G))

    n1 = len(part_nodes)
    n = len(G.nodes)
    # print(f"|Vq| = {n1}")
    # print(f"|V| = {n}")
    # print(f"|Vq|/|V| = {n1/n}")
    
    condac = nx.conductance(G, part_nodes)
    # print(f"Conductance equals to {condac}")
    # print(f'query nodes {part_nodes}')
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
        # print("Searching for the best balance accuracy...")
        # print(f"Current Balanced Accuracy for {mu_MS} is {ba_current}")
        # print(f"Best Balanced Accuracy is for {best_mu}, equal to {best_ba}")
        if((counter>10) or (best_ba==1.0)): break
    return best_mu

if __name__ == '__main__':

    #graph_names = ['ant', 'football', 'highschool', 'malaria', 'powerlaw_200_50_50', 'renyi_200_50', 'barabasi_200_50']
    
    dataset = sys.argv[1]
    percentage_lower_bound = int(sys.argv[2])
    percentage_upper_bound = int(sys.argv[3])
    per = float(sys.argv[4])
    edge_removal = float(sys.argv[5])
    graph_names = [dataset]

    standard_voting_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    neighborhood_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    initial_dict_standard = {threshold: [] for threshold in standard_voting_thresholds}
    initial_dict_neighborhood = {threshold: [] for threshold in neighborhood_thresholds}

    graphs = []
    use_global_mu = True
    for graph_name in graph_names:
        
        res_dict={}
        res_dict[graph_name] = {}
        best_mu = {}
        best_m_par = 0
        counter_m_par = 0
        folder_amount = 5
        for folder_no in range(folder_amount - 1, folder_amount):
            if use_global_mu and folder_no==0: continue
            # perc = [0.1, 0.2, 0.3]
            clcr = [i/10.0 for i in range(percentage_lower_bound, percentage_upper_bound)]
            if folder_no == 0: 
                best_mu[per] = {}
            res_dict[graph_name][int(per*100)] = {}
            for lr in clcr:   
                
                # Create dictionaries for standard voting
                standard_voting_balanced_accuracies = deepcopy(initial_dict_standard)
                standard_voting_accuracies = deepcopy(initial_dict_standard)
                standard_voting_recalls = deepcopy(initial_dict_standard)
                standard_voting_precisions = deepcopy(initial_dict_standard)
                standard_voting_f1s = deepcopy(initial_dict_standard)
                standard_voting_ged = deepcopy(initial_dict_standard)
                standard_voting_spectrum = deepcopy(initial_dict_standard)
                standard_voting_spectrum_diff = deepcopy(initial_dict_standard)

                # Create dictionaries for standard voting
                cc_standard_voting_balanced_accuracies = deepcopy(initial_dict_standard)
                cc_standard_voting_accuracies = deepcopy(initial_dict_standard)
                cc_standard_voting_recalls = deepcopy(initial_dict_standard)
                cc_standard_voting_precisions = deepcopy(initial_dict_standard)
                cc_standard_voting_f1s = deepcopy(initial_dict_standard)
                cc_standard_voting_ged = deepcopy(initial_dict_standard)
                cc_standard_voting_spectrum = deepcopy(initial_dict_standard)
                cc_standard_voting_spectrum_diff = deepcopy(initial_dict_standard)

                # Create dictionaries for neighborhood
                neighborhood_balanced_accuracies = deepcopy(initial_dict_neighborhood)
                neighborhood_accuracies = deepcopy(initial_dict_neighborhood)
                neighborhood_recalls = deepcopy(initial_dict_neighborhood)   
                neighborhood_precisions = deepcopy(initial_dict_neighborhood)
                neighborhood_f1s = deepcopy(initial_dict_neighborhood)
                neighborhood_ged = deepcopy(initial_dict_neighborhood)
                neighborhood_spectrum = deepcopy(initial_dict_neighborhood)
                neighborhood_spectrum_diff = deepcopy(initial_dict_neighborhood)

                # Create dictionaries for neighborhood
                cc_neighborhood_balanced_accuracies = deepcopy(initial_dict_neighborhood)
                cc_neighborhood_accuracies = deepcopy(initial_dict_neighborhood)
                cc_neighborhood_recalls = deepcopy(initial_dict_neighborhood)   
                cc_neighborhood_precisions = deepcopy(initial_dict_neighborhood)
                cc_neighborhood_f1s = deepcopy(initial_dict_neighborhood)
                cc_neighborhood_ged = deepcopy(initial_dict_neighborhood)
                cc_neighborhood_spectrum = deepcopy(initial_dict_neighborhood)
                cc_neighborhood_spectrum_diff = deepcopy(initial_dict_neighborhood)

                # Lists for original results
                og_balanced_accuracies = []
                og_accuracies = []
                og_recalls = []
                og_precisions = []
                og_f1s = []
                og_ged = []
                og_spectrum = []
                og_spectrum_diff = []

                conductances = []
                edge_removals = []
                all_votes = []
                ground_truth = []

                script_dir = os.path.dirname(__file__)
                rel_path = f'experiments_final_3/{graph_name}/{per}/{lr*100}'
                Path(rel_path).mkdir(parents=True, exist_ok=True)
                abs_file_path = os.path.join(script_dir, rel_path)
                if folder_no == 0: 
                    best_mu[per][lr] = 0
                ext = str(int(per*100))
                part_file = './data/'+graph_name+'/'+str(int(per*100))+'/'+str(folder_no)+'/'+graph_name+'_'+ext+'_nodes.txt'
                # print(f'Reading subgraph from {part_file}')
                query_nodes=np.loadtxt(part_file)
                # print(query_nodes)
                #print(part_nodes)
                ext = str(int(per*100))+"_"+str(int(lr*100))
                edgefile = './data/'+graph_name+'/'+str(int(per*100))+'/'+str(folder_no)+'/'+graph_name+'_'+ext+'.txt'
                if folder_no == 0:
                    # print('**** Looking for the best m ****')
                    best_mu[per][lr] = find_best_mu(edgefile, query_nodes)
                    best_m_par += best_mu[per][lr]
                    counter_m_par += 1
                else:
                    if use_global_mu:
                        standard_voting_results, neighborhood_results, condac, og_results, ref_spectrum, standard_voting_results_with_cardinality_constraint, neighborhood_results_with_cardinality_constraint, votes, v_gt = run_opt(edgefile,query_nodes, 0.2, standard_voting_thresholds, neighborhood_thresholds, edge_removal)
                        conductances.append(condac)
                        edge_removals.append(edge_removal)
                        all_votes.append(votes)
                        ground_truth.append(v_gt)
                        f = open(f'{abs_file_path}/ref_spectrum.txt', 'a+')
                        f.write(str(ref_spectrum))
                        # res_dict[graph_name][(int(per*100))][condac] = [acc, bal_acc, 0.2]
                        for result in standard_voting_results:
                            threshold = result["threshold"]
                            standard_voting_balanced_accuracies[threshold].append(result["balanced_acc"])   
                            standard_voting_accuracies[threshold].append(result["acc"])
                            standard_voting_recalls[threshold].append(result["recall"])
                            standard_voting_precisions[threshold].append(result["precision"])
                            standard_voting_f1s[threshold].append(result["f1"])
                            # standard_voting_ged[threshold].append(result["graph_edit_distance"])
                            standard_voting_spectrum[threshold].append(result["spectrum"])
                            standard_voting_spectrum_diff[threshold].append(result["spectrum_diff"])
                        for result in standard_voting_results_with_cardinality_constraint:
                            threshold = result["threshold"]
                            cc_standard_voting_balanced_accuracies[threshold].append(result["balanced_acc"])   
                            cc_standard_voting_accuracies[threshold].append(result["acc"])
                            cc_standard_voting_recalls[threshold].append(result["recall"])
                            cc_standard_voting_precisions[threshold].append(result["precision"])
                            cc_standard_voting_f1s[threshold].append(result["f1"])
                            # cc_standard_voting_ged[threshold].append(result["graph_edit_distance"])
                            cc_standard_voting_spectrum[threshold].append(result["spectrum"])
                            cc_standard_voting_spectrum_diff[threshold].append(result["spectrum_diff"])
                        for result in neighborhood_results:
                            threshold = result["threshold"]
                            neighborhood_balanced_accuracies[threshold].append(result["balanced_acc"])   
                            neighborhood_accuracies[threshold].append(result["acc"])
                            neighborhood_recalls[threshold].append(result["recall"])
                            neighborhood_precisions[threshold].append(result["precision"])
                            neighborhood_f1s[threshold].append(result["f1"])
                            # neighborhood_ged[threshold].append(result["graph_edit_distance"])
                            neighborhood_spectrum[threshold].append(result["spectrum"])
                            neighborhood_spectrum_diff[threshold].append(result["spectrum_diff"])
                        for result in neighborhood_results_with_cardinality_constraint:
                            threshold = result["threshold"]
                            cc_neighborhood_balanced_accuracies[threshold].append(result["balanced_acc"])   
                            cc_neighborhood_accuracies[threshold].append(result["acc"])
                            cc_neighborhood_recalls[threshold].append(result["recall"])
                            cc_neighborhood_precisions[threshold].append(result["precision"])
                            cc_neighborhood_f1s[threshold].append(result["f1"])
                            # cc_neighborhood_ged[threshold].append(result["graph_edit_distance"])
                            cc_neighborhood_spectrum[threshold].append(result["spectrum"])
                            cc_neighborhood_spectrum_diff[threshold].append(result["spectrum_diff"])
                        og_balanced_accuracies.append(og_results["balanced_acc"])
                        og_accuracies.append(og_results["acc"])
                        og_precisions.append(og_results["precision"])
                        og_recalls.append(og_results["recall"])
                        og_f1s.append(og_results["f1"])
                        # og_ged.append(og_results["graph_edit_distance"])
                        og_spectrum.append(og_results["spectrum"])
                        og_spectrum_diff.append(og_results["spectrum_diff"])

                    else:
                        standard_voting_results, neighborhood_results, condac, og_results, ref_spectrum, standard_voting_results_with_cardinality_constraint, neighborhood_results_with_cardinality_constraint, votes, v_gt = run_opt(edgefile,query_nodes, best_mu[per][lr], standard_voting_thresholds, neighborhood_thresholds, edge_removal)
                        conductances.append(condac)
                        edge_removals.append(edge_removal)
                        all_votes.append(votes)
                        ground_truth.append(v_gt)
                        # res_dict[graph_name][(int(per*100))][condac] = [acc, bal_acc, 0.2]
                        for result in standard_voting_results:
                            threshold = result["threshold"]
                            standard_voting_balanced_accuracies[threshold].append(result["balanced_acc"])   
                            standard_voting_accuracies[threshold].append(result["acc"])
                            standard_voting_recalls[threshold].append(result["recall"])
                            standard_voting_precisions[threshold].append(result["precision"])
                            standard_voting_f1s[threshold].append(result["f1"])
                            # standard_voting_ged[threshold].append(result["graph_edit_distance"])
                        for result in neighborhood_results:
                            threshold = result["threshold"]
                            neighborhood_balanced_accuracies[threshold].append(result["balanced_acc"])   
                            neighborhood_accuracies[threshold].append(result["acc"])
                            neighborhood_recalls[threshold].append(result["recall"])
                            neighborhood_precisions[threshold].append(result["precision"])
                            neighborhood_f1s[threshold].append(result["f1"])
                            # neighborhood_ged[threshold].append(result["graph_edit_distance"])
                        og_balanced_accuracies.append(og_results["balanced_acc"])
                        og_accuracies.append(og_results["acc"])
                        og_precisions.append(og_results["precision"])
                        og_recalls.append(og_results["recall"])
                        og_f1s.append(og_results["f1"])
                        # og_ged.append(og_results["graph_edit_distance"])

                       # Write results for standard voting 
                rel_path = f'experiments_final_3/{graph_name}/{per}/{lr*100}'
                Path(rel_path).mkdir(parents=True, exist_ok=True)
                script_dir = os.path.dirname(__file__)
                abs_file_path = os.path.join(script_dir, rel_path)
                
                f = open(f'{abs_file_path}/conductance.txt', 'a+')
                f.write(str(conductances))

                f = open(f'{abs_file_path}/edge_removal.txt', 'a+')
                f.write(str(edge_removals))

                f = open(f'{abs_file_path}/votes', 'a+')
                f.write(str(all_votes))

                f = open(f'{abs_file_path}/ground_truth', 'a+')
                f.write(str(ground_truth))

                # Writing data for standard voting
                for threshold, values in standard_voting_balanced_accuracies.items():
                    f = open(f'{abs_file_path}/balanced_accuracy_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in standard_voting_accuracies.items():
                    f = open(f'{abs_file_path}/accuracy_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in standard_voting_recalls.items():
                    f = open(f'{abs_file_path}/recall_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in standard_voting_precisions.items():
                    f = open(f'{abs_file_path}/precision_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in standard_voting_f1s.items():
                    f = open(f'{abs_file_path}/f1_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in standard_voting_ged.items():
                    f = open(f'{abs_file_path}/ged_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in standard_voting_spectrum.items():
                    f = open(f'{abs_file_path}/spectrum_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in standard_voting_spectrum_diff.items():
                    f = open(f'{abs_file_path}/spectrum_diff_{threshold}.txt', 'a+')
                    f.write(str(values))

                # Writing data for standard voting with cardinality constraint
                for threshold, values in cc_standard_voting_balanced_accuracies.items():
                    f = open(f'{abs_file_path}/cc_balanced_accuracy_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in cc_standard_voting_accuracies.items():
                    f = open(f'{abs_file_path}/cc_accuracy_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in cc_standard_voting_recalls.items():
                    f = open(f'{abs_file_path}/cc_recall_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in cc_standard_voting_precisions.items():
                    f = open(f'{abs_file_path}/cc_precision_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in cc_standard_voting_f1s.items():
                    f = open(f'{abs_file_path}/cc_f1_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in cc_standard_voting_ged.items():
                    f = open(f'{abs_file_path}/cc_ged_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in cc_standard_voting_spectrum.items():
                    f = open(f'{abs_file_path}/cc_spectrum_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in cc_standard_voting_spectrum_diff.items():
                    f = open(f'{abs_file_path}/cc_spectrum_diff_{threshold}.txt', 'a+')
                    f.write(str(values))

                # Write results for neighborhood
                for threshold, values in neighborhood_balanced_accuracies.items():
                    f = open(f'{abs_file_path}/n_balanced_accuracy_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in neighborhood_accuracies.items():
                    f = open(f'{abs_file_path}/n_accuracy_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in neighborhood_recalls.items():
                    f = open(f'{abs_file_path}/n_recall_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in neighborhood_precisions.items():
                    f = open(f'{abs_file_path}/n_precision_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in neighborhood_f1s.items():
                    f = open(f'{abs_file_path}/n_f1_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in neighborhood_ged.items():
                    f = open(f'{abs_file_path}/n_ged_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in neighborhood_spectrum.items():
                    f = open(f'{abs_file_path}/n_spectrum_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in neighborhood_spectrum_diff.items():
                    f = open(f'{abs_file_path}/n_spectrum_diff_{threshold}.txt', 'a+')
                    f.write(str(values))

                # Write results for neighborhood with cardinality constraint
                for threshold, values in cc_neighborhood_balanced_accuracies.items():
                    f = open(f'{abs_file_path}/cc_n_balanced_accuracy_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in cc_neighborhood_accuracies.items():
                    f = open(f'{abs_file_path}/cc_n_accuracy_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in cc_neighborhood_recalls.items():
                    f = open(f'{abs_file_path}/cc_n_recall_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in cc_neighborhood_precisions.items():
                    f = open(f'{abs_file_path}/cc_n_precision_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in cc_neighborhood_f1s.items():
                    f = open(f'{abs_file_path}/cc_n_f1_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in cc_neighborhood_ged.items():
                    f = open(f'{abs_file_path}/cc_n_ged_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in cc_neighborhood_spectrum.items():
                    f = open(f'{abs_file_path}/cc_n_spectrum_{threshold}.txt', 'a+')
                    f.write(str(values))

                for threshold, values in cc_neighborhood_spectrum_diff.items():
                    f = open(f'{abs_file_path}/cc_n_spectrum_diff_{threshold}.txt', 'a+')
                    f.write(str(values))

                # Write for original results
                f = open(f'{abs_file_path}/og_balanced_accuracy.txt', 'a+')
                f.write(str(og_balanced_accuracies))

                f = open(f'{abs_file_path}/og_accuracy.txt', 'a+')
                f.write(str(og_accuracies))

                f = open(f'{abs_file_path}/og_recall.txt', 'a+')
                f.write(str(og_recalls))

                f = open(f'{abs_file_path}/og_precision.txt', 'a+')
                f.write(str(og_precisions))

                f = open(f'{abs_file_path}/og_f1.txt', 'a+')
                f.write(str(og_f1s))

                f = open(f'{abs_file_path}/og_ged.txt', 'a+')
                f.write(str(og_ged))

                f = open(f'{abs_file_path}/og_spectrum.txt', 'a+')
                f.write(str(og_spectrum))

                f = open(f'{abs_file_path}/og_spectrum_diff.txt', 'a+')
                f.write(str(og_spectrum_diff))
