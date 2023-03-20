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
from problem.spectral_subgraph_localization import SubgraphIsomorphismSolver, VotingSubgraphIsomorpishmSolver, VotingSubgraphIsomorpishmSolver
import pickle
import sys



from sklearn.metrics import balanced_accuracy_score
def balanced_acc(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)

def accur(y_true, y_pred):
    counter = 0
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == 0:
            counter += 1
            if y_true[i] == y_pred[i]:
                correct += 1
    return 1.0*correct/counter
        
    

def run_opt(edgefile,part_nodes, mu=1):

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
    print(part_nodes)


    lowest_degree = np.inf
    sub_G = nx.subgraph(G, part_nodes)
    for degree_view in nx.degree(sub_G):
        lowest_degree = min(lowest_degree,degree_view[1])
    print("removing edges:", list(n for n in G.nodes if nx.degree(G,n)<lowest_degree))
    G.remove_nodes_from(list(n for n in G.nodes if nx.degree(G,n)<lowest_degree))

    if condac == 0:
        print("Conductance was 0, so we skip (the algorithm already works well on these graphs)")
        return (0, 0, 0)

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
    mu_MS_list = np.linspace(10*1e0,10*1e2,2)

    threshold_E = False
    if True:
        problem_params = {'mu_spectral': 1,
                          'mu_l21': 0,
                          'mu_MS': mu,# / (c ** 2), #The μ regularizer eq. 11
                          'mu_split': 0,
                          'mu_trace': 0.0,
                          'trace_val': 0,
                          'weighted_flag': False
                          }
        subgraph_isomorphism_solver.set_problem_params(problem_params)
        v, E = \
            subgraph_isomorphism_solver.solve(max_outer_iters=3,max_inner_iters=500, show_iter=10000, verbose=False)

        v_binary, E_binary = subgraph_isomorphism_solver.threshold(v_np=v.detach().numpy())
        v_bin_spectral, _ = subgraph_isomorphism_solver.threshold(v_np=v.detach().numpy(), threshold_algo="spectral")
        v_bin_smallest, _ = subgraph_isomorphism_solver.threshold(v_np=v.detach().numpy(), threshold_algo="smallest")
        gt_inidicator = v_gt
        gt_inidicator[gt_inidicator>0]=1 

        original_accuracy = accur(v_gt, v_binary.clone().detach().numpy())
        original_balanced = balanced_acc(v_gt, v_binary.clone().detach().numpy())
        print("Original balanced accuracy:", original_balanced)
        original_balanced_spectral = balanced_acc(v_gt, v_bin_spectral.clone().detach().numpy())
        print("Original balanced with spectral threshold algo:", original_balanced_spectral)
        original_balanced_smallest = balanced_acc(v_gt, v_bin_smallest.clone().detach().numpy())
        print("Original balanced with smallest threshold algo:", original_balanced_smallest)

        if original_balanced > 0.9:
            print("Original accuracy was already high. Skipping.")
            return (original_accuracy, original_balanced, condac)

        random_solver = VotingSubgraphIsomorpishmSolver(A, ref_spectrum, problem_params, solver_params, v_gt, original_balanced)
        v_randomized, _, solutions = random_solver.solve(max_outer_iters=3,max_inner_iters=500, show_iter=10000, verbose=False)

        v_bin_spectral, _ = subgraph_isomorphism_solver.threshold(v_np=v.detach().numpy(), threshold_algo="spectral")
        v_bin_smallest, _ = subgraph_isomorphism_solver.threshold(v_np=v.detach().numpy(), threshold_algo="smallest")
        gt_inidicator = v_gt
        gt_inidicator[gt_inidicator>0]=1 

        original_accuracy = accur(v_gt, v_binary.clone().detach().numpy())
        original_balanced = balanced_acc(v_gt, v_binary.clone().detach().numpy())
        print("Original balanced accuracy:", original_balanced)
        original_balanced_spectral = balanced_acc(v_gt, v_bin_spectral.clone().detach().numpy())
        print("Original balanced with spectral threshold algo:", original_balanced_spectral)
        original_balanced_smallest = balanced_acc(v_gt, v_bin_smallest.clone().detach().numpy())
        print("Original balanced with smallest threshold algo:", original_balanced_smallest)

        if original_balanced > 0.9:
            print("Original accuracy was already high. Skipping.")
            return (original_accuracy, original_balanced, condac)

        random_solver = VotingSubgraphIsomorpishmSolver(A, ref_spectrum, problem_params, solver_params, v_gt, original_balanced)
        v_randomized, _, solutions = random_solver.solve(max_outer_iters=3,max_inner_iters=500, show_iter=10000, verbose=False)


        if threshold_E:
            v_clustered, E_clustered = subgraph_isomorphism_solver.threshold(v_np = v.detach().numpy())
            subgraph_isomorphism_solver.set_init(E0 = E_clustered, v0 = v)



    #pause(1)
    v_binary, E_binary = subgraph_isomorphism_solver.threshold(v_np=v.detach().numpy())
    idx_smallest=np.argsort(v.detach().numpy())[:n1]
    v_smallest=np.ones((n,1))
    for i in range(n):
        if i in idx_smallest:
            v_smallest[i]=0

    randomized_accuracy, randomized_balanced = accur(v_gt, v_randomized.clone().detach().numpy()), balanced_acc(v_gt, v_randomized.clone().detach().numpy())

    print("Accuracy, original vs randomized", original_accuracy, randomized_accuracy)
    print("balanced Accuracy, original vs randomized", original_balanced, randomized_balanced)  

    nodes_in_res = count_nodes(v_binary)
    print("nodes in original solution", nodes_in_res)

    nodes_in_random_res = count_nodes(v_randomized)
    print("nodes in voting solution", nodes_in_random_res)

    # Returning original accuracy
    return (accur(v_gt, v_binary.clone().detach().numpy()), balanced_acc(v_gt, v_binary.clone().detach().numpy()), condac)

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
    # graph_names = ['ant', 'football', 'highschool', 'malaria', 'powerlaw_200_50_50', 'renyi_200_50', 'barabasi_200_50']
    graph_names = ['highschool']
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
            clcr = [i/10.0 for i in range(folderAmount)]
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
                            (acc, bal_acc, condac) =run_opt(edgefile,query_nodes, 0.2)
                            print(f"Balanced Accuracy {bal_acc}")
                            print(f"Accuracy {acc}\n")
                            res_dict[graph_name][(int(per*100))][condac] = [acc, bal_acc, 0.2]
                        else:
                            (acc, bal_acc, condac) =run_opt(edgefile,query_nodes, best_mu[per][lr])
                            print(f"Balanced Accuracy {bal_acc}")
                            print(f"Accuracy {acc}\n")
                            res_dict[graph_name][(int(per*100))][condac] = [acc, bal_acc]
            print(res_dict)


            if(True):
                name_to_save = graph_name+"_"+str(folder_no)
                directo = './pkl_results/'+graph_name+'/'+name_to_save+'.pkl'
                import os
                if not os.path.exists('./pkl_results/'+graph_name):
                        os.makedirs('./pkl_results/'+graph_name)
                with open(directo, 'wb') as f:
                    pickle.dump(res_dict, f)
            #sys.exit()


