import numpy as np
import scipy as sci
import networkx as nx
import time
import lapjv
import os
#import munkres

from sklearn.neighbors import NearestNeighbors
#np.set_printoptions(precision=4)

#folder="C:/Users/Judith/Documents/MATLAB/compare_matlab_python/"


# 	#noise_level= [1]
# 	scores=np.zeros([5,reps])
# 	for i in noise_level:
# 		for j in range(1,reps+1):
# 			print('Noise level %d, round %d' %(i,j))

def main():

    functional_maps(30,20,'arenas')
    functional_maps(30, 20, 'facebook')
    functional_maps(30, 20, 'CA-HepTh')
    functional_maps(30, 20, 'CA-AstroPh')

   # print("\n")

def functional_maps(q,k,graph_name):
    variant='nn'
    filename='functionalMaps_nn.py'
    #q = 100
    #k = 20


    t = np.linspace(1, 100, q)
    reps =5
    noise_level = [1, 2, 3, 4, 5,6,7,8,9,10];



    #laplacian eigenvector corresponding function stuff for graph 1


    edge_list_G1 = 'permutations/'+ graph_name +'/'+ graph_name +'_orig.txt'
    #edge_list_G1 = 'permutations/arenas/arenas_orig.txt'
    A1 = edgelist_to_adjmatrix(edge_list_G1)
    n = np.shape(A1)[0]

    #D1=np.empty([n,n])
    #V1=np.empty([n])

    exists=os.path.isfile('eigens/'+ graph_name +'/'+ graph_name+ '_evectors_orig.npy')
    if(exists):
        #print("found")
        D1=np.load('eigens/'+ graph_name +'/'+ graph_name+ '_evalues_orig.npy')
        V1=np.load('eigens/'+ graph_name +'/'+ graph_name+ '_evectors_orig.npy')
    else:
        #print("not found")
        D1, V1 = decompose_laplacian(A1)
        np.save('eigens/'+ graph_name +'/'+ graph_name+ '_evalues_orig.npy', D1)
        np.save('eigens/'+ graph_name +'/'+ graph_name+ '_evectors_orig.npy', V1)

    #D1, V1 = decompose_laplacian(A1)


    Cor1 = calc_corresponding_functions(n, q, t, D1, V1)
    A = calc_coefficient_matrix(Cor1, V1, k, q)
    scores = np.zeros([10, reps])
    times =  np.zeros([10, reps])
    for i in noise_level:
        for j in range(1,reps+1):

            start_time = time.time()
            print('Noise level %d, round %d' %(i,j))
            print('Current graph: '+graph_name+', current file: '+filename)
            edge_list_G2 = 'permutations/'+graph_name+'/noise_level_'+str(i)+'/edges_'+str(j)+'.txt'
            #edge_list_G2 = 'permutations/arenas/noise_level_' + str(i) + '/edges_' + str(j) + '.txt'
            A2 = edgelist_to_adjmatrix(edge_list_G2)

            #D2 = np.empty([n, n])
            #V2 = np.empty([n])
            exists = os.path.isfile('eigens/'+ graph_name +'/'+  str(i) + '/evectors_' + str(j) + '.npy')
            if (exists):
                #print("found")
                D2 = np.load('eigens/'+ graph_name +'/'+  str(i) + '/evectors_' + str(j) + '.npy')
                V2 = np.load('eigens/'+ graph_name +'/'+ str(i) + '/evalues_' + str(j) + '.npy')
            else:
                #print("not found")
                D2, V2 = decompose_laplacian(A2)
                np.save('eigens/'+ graph_name +'/'+  str(i) + '/evectors_' + str(j) + '.npy', D2)
                np.save('eigens/'+ graph_name  +'/'+  str(i) + '/evalues_' + str(j) + '.npy', V2)

            #D2, V2 = decompose_laplacian(A2)
            #print(np.shape(D2))
            #print(np.shape(V2))
            Cor2 = calc_corresponding_functions(n, q, t, D2, V2)
            B = calc_coefficient_matrix(Cor2, V2, k, q)



            C=calc_correspondence_matrix(A,B,k)

            G1_emb = C @ V1[:, 0: k].T;

            G2_emb = V2[:, 0: k].T;


            #hungarian matching
            #matching = hungarian_matching(G1_emb, G2_emb)
            #nearest neighbor matching
            matching = nearest_neighbor_matching(G1_emb, G2_emb)

            #print(matching[matching[:, 0].argsort()])

            #print(matching)
            #icp
           # matching= iterative_closest_point(V1, V2, C, 2, k)
           # matching =greedyNN(G1_emb, G2_emb)
           # matching=sort_greedy(G1_emb, G2_emb)
            matching = dict(matching.astype(int))
           # print(matching)
            gt_file = 'permutations/'+graph_name+'/noise_level_'+str(i)+'/gt_'+str(j)+'.txt'
            #gt_file = 'permutations/arenas/noise_level_' + str(i) + '/gt_' + str(j) + '.txt'

            data = np.loadtxt(gt_file, delimiter=" ")

#            data = data - np.ones([n, 2])
           # data=np.c_[data[:,1],data[:,0]]

            data.astype(int)

            gt=dict(data)

            acc=eval_matching(matching ,gt)
            scores[i - 1, j - 1] = acc
            cur_time=time.time()-start_time
            times[i - 1, j - 1] = cur_time

            print("accuracy: %f, time: %f" % (acc, cur_time))
            acc_file='gasp_variants/'+graph_name+'_accs_'+variant
            time_file = 'gasp_variants/'+graph_name+'_times_'+variant

            #acc_file = 'results/accs_q' + str(q) + '_k' + str(k)
            #time_file = 'results/times_q' + str(q) + '_k' + str(k)

            np.save(acc_file,scores)
            np.save(time_file, times)

            print("\n")

    return scores, times
    #sci.io.savemat('C:/Users/Judith/Documents/MATLAB/py_ours_scores.mat', {'scores':scores})

def edgelist_to_adjmatrix(edgeList_file):

    edge_list = np.loadtxt(edgeList_file, usecols=range(2))

    n = int(np.amax(edge_list)+1)
   # print(n)

    e = np.shape(edge_list)[0]


    a = np.zeros((n, n))

    # make adjacency matrix A1

    for i in range(0, e):
        n1 = int(edge_list[i, 0])# - 1

        n2 = int(edge_list[i, 1])#- 1

        a[n1, n2] = 1.0
        a[n2, n1] = 1.0

    return a


def decompose_laplacian(A):

    #  adjacency matrix

    Deg = np.diag((np.sum(A, axis=1)))

    n = np.shape(Deg)[0]

    Deg=sci.linalg.fractional_matrix_power(Deg, -0.5)

    L  = np.identity(n) - Deg @ A @ Deg
   # print((sci.fractional_matrix_power(Deg, -0.5) * A * sci.fractional_matrix_power(Deg, -0.5)))
    # '[V1, D1] = eig(L1);

    D, V = np.linalg.eigh(L)

    return [D, V]


def calc_corresponding_functions(n, q, t, d, V):

    # corresponding functions are the heat kernel diagonals in each time step
    # t= time steps, d= eigenvalues, V= eigenvectors, n= number of nodes, q= number of corresponding functions
    t = t[:, np.newaxis]
    d = d[:, np.newaxis]

    V_square = np.square(V)

    time_and_eigv = np.dot((d), np.transpose(t))

    time_and_eigv = np.exp(-1*time_and_eigv)

    Cores=np.dot(V_square, time_and_eigv)

    return Cores


def calc_coefficient_matrix(Corr, V, k, q):
    coefficient_matrix = np.linalg.lstsq(V[:,0:k],Corr,rcond=None)
    #print(type(coefficient_matrix))
    return coefficient_matrix[0]

def calc_correspondence_matrix(A, B, k):
    C = np.zeros([k,k])
    At = A.T
    Bt = B.T

    for i in range(0,k):
        C[i, i] = np.linalg.lstsq(Bt[:,i].reshape(-1,1), At[:,i].reshape(-1,1),rcond=None)[0]

    return C

def nearest_neighbor_matching(G1_emb, G2_emb):
    n= np.shape(G1_emb)[1]
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(G1_emb.T)
    distances, indices = nbrs.kneighbors(G2_emb.T)
    print("dist shape: " +str(np.shape(distances)[0])+ ', '+str(np.shape(distances)[1]))
    indices=np.c_[np.linspace(0, n-1, n).astype(int), indices.astype(int)]
    return indices
#
def hungarian_matching(G1_emb, G2_emb):
    print('hungarian_matching: calculating distance matrix')

    dist = sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)
    n = np.shape(dist)[0]
    # print(np.shape(dist))
    print('hungarian_matching: calculating matching')
    cols, rows, _ = lapjv.lapjv(dist)
    print('fertig')
    matching = np.c_[cols, np.linspace(0, n-1, n).astype(int)]
    matching=matching[matching[:,0].argsort()]
    return matching.astype(int)

def iterative_closest_point(V1, V2, C, it,k):
    G1= V1[:, 0: k].T
    G2_emb = V2[:, 0: k].T
    n = np.shape(G2_emb)[1]

    for i in range(0,it):

        print('icp iteration '+str(i))
        G1_emb=C@V1[:,0:k].T
       # print('calculating hungarian in icp')

        #M=hungarian_matching(G1_emb,G2_emb)
       # M=greedyNN(G1_emb,G2_emb)
        M=sort_greedy(G1_emb, G2_emb)

        G2_cur=np.zeros([k,n])
       ## print('finding nearest neighbors in eigenvector matrix icp')
        for j in range(0,n):

            G2idx = M[j, 1]

            G2_cur[:, G2idx]=G2_emb[:, j]
       ## print('calculating correspondence matrix in icp')
        C=calc_correspondence_matrix(G1,G2_cur,k)
       # print('calculated correspondence matrix in icp')
       ## print('\n')
    G1_emb = C@V1[:,0:k].T


   # M = hungarian_matching(G1_emb,G2_emb)
   # M = greedyNN(G1_emb, G2_emb)
    M = sort_greedy(G1_emb, G2_emb)

    return M

def greedyNN(G1_emb, G2_emb):
    print('greedyNN: calculating distance matrix')

    dist = sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)
    n = np.shape(dist)[0]
    print(np.shape(dist))
    print('greedyNN: calculating matching')
    idx = np.argsort(dist, axis=0)
    matching=np.ones([n,1])*(n+1)
    for i in range(0,n):
        matched=False
        cur_idx=0
        while(not matched):
           #print([cur_idx,i])
           if(not idx[cur_idx,i] in matching):
               matching[i,0]=idx[cur_idx,i]

               matched=True
           else:
               cur_idx += 1
               #print(cur_idx)

    matching = np.c_[np.linspace(0, n-1, n).astype(int),matching]
    return matching.astype(int)

def sort_greedy(G1_emb, G2_emb):
    print('sortGreedy: calculating distance matrix')

    dist = sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)
    n = np.shape(dist)[0]
    # print(np.shape(dist))
    print('sortGreedy: calculating matching')
    dist_platt=np.ndarray.flatten(dist)
    idx = np.argsort(dist_platt)#
    k=idx//n
    r=idx%n
    idx_matr=np.c_[k,r]
   # print(idx_matr)
    G1_elements=set()
    G2_elements=set()
    i=0
    j=0
    matching=np.ones([n,2])*(n+1)
    while(len(G1_elements)<n):
        if (not idx_matr[i,0] in G1_elements) and (not idx_matr[i,1] in G2_elements):
            #print(idx_matr[i,:])
            matching[j,:]=idx_matr[i,:]

            G1_elements.add(idx_matr[i,0])
            G2_elements.add(idx_matr[i,1])
            j+=1
            #print(len(G1_elements))


        i+=1

   # print(idx)
    matching = np.c_[matching[:,1], matching[:,0]]
    matching = matching[matching[:, 0].argsort()]
    return matching.astype(int)

def eval_matching(matching, gt):

    n=float(len(gt))
    acc=0.0
    for i in matching:
        if i in gt and matching[i] == gt[i]:
                acc+=1.0
   # print(acc/n)
    return acc/n

def read_regal_matrix(file):

    nx_graph = nx.read_edgelist(file, nodetype=int, comments="%")
    A = nx.adjacency_matrix(nx_graph)
    n=int(np.shape(A)[0]/2)
    A1 = A[0:n,0:n]
    A2 = A[n:2*n,n:2*n]
    return A1.todense(), A2.todense()



if __name__ == '__main__':
    main()
