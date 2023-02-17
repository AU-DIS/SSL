from ast import And
from tkinter import N
import numpy as np
import networkx as nx
import sys
import matplotlib.pyplot as plt
import random
sys.path.insert(0, '..')
import os

def edgelist_to_adjmatrix(edgeList_file):
    edge_list = np.loadtxt(edgeList_file, usecols=range(2))

    n = int(np.amax(edge_list) + 1)
    # n = int(np.amax(edge_list))
    # print(n)

    e = np.shape(edge_list)[0]

    a = np.zeros((n, n))

    # make adjacency matrix A1

    for i in range(0, e):
        n1 = int(edge_list[i, 0])  # - 1

        n2 = int(edge_list[i, 1])  # - 1

        a[n1, n2] = 1.0
        a[n2, n1] = 1.0

    return a

def random_number(max_num):
    return random.randint(0,max_num-1)

def random_numbers(max_num, total_el):
    my_list = []
    while len(my_list) < total_el:
        rn = random_number(max_num)
        if rn in my_list:
            continue
        my_list.append(rn)
    return my_list

def select_connected_subgraph(G, num_of_nodes):
    rest_nodes = [int(i) for i in G.nodes()]
    query_nodes = []
    is_con = False
    while is_con == False:
        rest_nodes = [int(i) for i in G.nodes()]
        query_nodes = []
        cand_neighs = []
        for i in range(num_of_nodes):
            if len(cand_neighs) == 0:
                query_node = (random_number(len(rest_nodes)))
                query_nodes.append(query_node)
                if query_node in rest_nodes:
                    rest_nodes.remove(query_node)
                for neigh in G.neighbors(query_node):
                    cand_neighs.append(neigh)
            else:
                query_node = cand_neighs[(random_number(len(cand_neighs)))]
                cand_neighs.remove(query_node)
                query_nodes.append(query_node)
                rest_nodes.remove(query_node)
                for neigh in G.neighbors(query_node):
                    if neigh in cand_neighs: continue
                    if neigh in query_nodes: continue
                    cand_neighs.append(neigh)
        #print(check_connectivity(G.subgraph(query_nodes)), ' --- ',  check_connectivity(G.subgraph(rest_nodes)))
        is_con = check_connectivity(G.subgraph(query_nodes)) and check_connectivity(G.subgraph(rest_nodes))
    return query_nodes
            
def find_subgraph(G, num_nodes, unweighted = False):
    current_node = random_number(G.number_of_nodes())
    query_nodes = []
    pool_of_numbers = []
    query_nodes.append(current_node)
    for neigh in G.neighbors(current_node):
        pool_of_numbers.append(neigh)
    for _ in range(num_nodes - 1):
        current_node = pool_of_numbers[random_number(len(pool_of_numbers))]
        query_nodes.append(current_node)
        for neigh in G.neighbors(current_node):
            if(unweighted):
                if(neigh not in pool_of_numbers): pool_of_numbers.append(neigh)
            else:
                pool_of_numbers.append(neigh)
        pool_of_numbers = list(filter(lambda score: score != current_node, pool_of_numbers))
        #print(pool_of_numbers)

    return query_nodes
 
def add_edges(G, query_nodes, num_of_new_edges):
    for q_node in query_nodes:
        for node in G.nodes():
            if node not in query_nodes:
                if(G.has_edge(node, q_node) == False):
                    print(f'adding edge ({q_node}, {node})')
                    G.add_edge(q_node, node)
                    num_of_new_edges -= 1
                    if num_of_new_edges == 0:
                        return G
     

def write_list_in_file(results, graph_name, ext, folder_no = None):
    if folder_no is None:
        print('../MyDatasets/'+graph_name+'/'+graph_name+'_'+ext+'.txt')
        with open('../MyDatasets/'+graph_name+'/'+graph_name+'_'+ext+'.txt', 'w') as fp:
            for item in results:
                # write each item on a new line
                fp.write("%s\n" % item)
            print('Done')
    else:
        print('')
        if not os.path.exists('../MyDatasets/'+graph_name+'/'+str(folder_no)):
            os.makedirs('../MyDatasets/'+graph_name+'/'+str(folder_no))
        print(f'Writing in ', '../MyDatasets/'+graph_name+'/'+str(folder_no)+'/'+graph_name+'_'+ext+'.txt')
        with open('../MyDatasets/'+graph_name+'/'+str(folder_no)+'/'+graph_name+'_'+ext+'.txt', 'w') as fp:
            for item in results:
                fp.write("%s\n" % item)
        print('Done')

def write_graph_in_file(G, graph_name, ext, folder_no = None):
    if folder_no is None:
        edgefile = '../MyDatasets/'+graph_name+'/'+graph_name+'_'+ext+'.txt'
    else:
        if not os.path.exists('../MyDatasets/'+graph_name+'/'+str(folder_no)):
            os.makedirs('../MyDatasets/'+graph_name+'/'+str(folder_no))
        edgefile = '../MyDatasets/'+graph_name+'/'+str(folder_no)+'/'+graph_name+'_'+ext+'.txt'
    nx.write_edgelist(G, edgefile, data=False)    
    
def read_graph(edgefile):
    A = edgelist_to_adjmatrix(edgefile)
    G = nx.from_numpy_matrix(A)
    return G

def split_graph(G, perc, use_clique = False):
    n = G.number_of_nodes()
    n_sub = int(perc*n)
    is_con = False
    while is_con == False:
        #query_nodes = random_numbers(n, n_sub)
        query_nodes = select_connected_subgraph(G, n_sub)
        sub_G = nx.subgraph(G, query_nodes)
        rest_nodes = [node for node in range(n) if node not in query_nodes]
        rest_G = nx.subgraph(G, rest_nodes)
        #if use_clique:
            #G_new = make_it_clique(G, query_nodes)
        is_con = check_connectivity(sub_G) and check_connectivity(rest_G)
        #if use_clique and is_con:
           #G = G_new
    #nx.draw_networkx(sub_G)
    #plt.show()
    my_dict = {}
    my_dict['n'] = n
    my_dict['m'] = G.number_of_edges()
    
    my_dict['nr'] = n - n_sub
    my_dict['mr'] = rest_G.number_of_edges()
    
    my_dict['nq'] = n_sub
    my_dict['mq'] = sub_G.number_of_edges()
    
    my_dict['query'] = query_nodes
    my_dict['rest'] = rest_nodes
    return my_dict


def check_connectivity(G):
    return nx.is_connected(G)

def keep_edges(G, my_dict, perc_Q_to_Q, perc_Q_to_no_Q, use_clique = False):
    if (perc_Q_to_Q == 1) and (perc_Q_to_no_Q == 1):
        return G
    is_con = False
    while is_con == False:
        G_new = G.copy()
        #print('------ ', use_clique)
        
        query_nodes = my_dict['query']
        counter = nx.subgraph(G, my_dict['query']).number_of_edges()#my_dict['mq']
        mq_new = int(counter * perc_Q_to_Q)
        #print(f'--------> From {counter} to {mq_new}')
        for q_node1 in query_nodes:
            for q_node2 in query_nodes:
                if counter == mq_new: break
                if(G_new.has_edge(q_node1, q_node2)):
                    G_new.remove_edge(q_node1, q_node2)
                    counter -= 1
                    #print(f'counter {counter} -> mqnew {mq_new}')
                    if counter == mq_new: break
        is_con = check_connectivity(nx.subgraph(G_new, query_nodes))
        #if(use_clique):
            #is_con = True
            #continue
        if(is_con == False): 
            continue
            
        mqnq_new = int((my_dict['m'] - my_dict['mq'] - my_dict['mr'])  * perc_Q_to_no_Q)
        counter = my_dict['m'] - my_dict['mq'] - my_dict['mr']
        for q_node in query_nodes:
            for node in range(my_dict['n']):
                if counter == mqnq_new: break
                if node in query_nodes: continue
                if(G_new.has_edge(q_node, node)):
                    G_new.remove_edge(q_node, node)
                    counter -= 1
                    if counter == mqnq_new: break
    return G_new



def make_data(graph_name, perc, comb = [[0.5, 0.5]], clcr= None, folder_no = None, use_clique = False):
    edgefile = '../Mydatasets/'+graph_name+'/'+graph_name+'.txt'
    G = read_graph(edgefile)
    G = nx.subgraph(G, max(nx.connected_components(G), key=len))
    if use_clique:
        G = nx.complement(G)
    print(f'G has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges')
    for per in perc:
        print(f'Keep the {per*100} as query nodes')
        my_dict =  split_graph(G, per, use_clique)   
        if use_clique:
            write_list_in_file(my_dict['query'], graph_name, "query_"+str(int(per*100))+"_clique", folder_no)  
        else:
            write_list_in_file(my_dict['query'], graph_name, "query_"+str(int(per*100)), folder_no)   
        print(f'clcr {clcr} ')
        if clcr is not None:
            comb = []
            cl = clcr[0]
            print(f'cl {cl} ')
            cr = clcr[1]
            print(f'cr {cr} ')
            for l in cl:
                for r in cr:
                    comb.append([l, r])
        print(f'================   Combinations are {comb}')
        for c in comb:
            perc_Q_to_Q = c[0]
            perc_Q_to_no_Q = c[1]
            #if use_clique:
                #G = make_it_clique(G, my_dict['query'])
            mqnew = nx.subgraph(G, my_dict['query']).number_of_edges()
            print(f'BEFORE Inside the query subgraph are: {mqnew}')
            G_new = keep_edges(G, my_dict, perc_Q_to_Q, perc_Q_to_no_Q, use_clique)
            print(f'G_new total edges are: {G_new.number_of_edges()}')
            mqnew = nx.subgraph(G_new, my_dict['query']).number_of_edges()
            print(f'Inside the query subgraph are: {mqnew}')
            mrnew = nx.subgraph(G_new, my_dict['rest']).number_of_edges()
            print(f'From the query subgraph to rest are: {G_new.number_of_edges() - mrnew - mqnew}')
            if use_clique:
                ext = str(perc_Q_to_Q) + "_" + str(perc_Q_to_no_Q)+ "_" + str(int(per*100))+ "_clique"
            else: 
                ext = str(perc_Q_to_Q) + "_" + str(perc_Q_to_no_Q)+ "_" + str(int(per*100))
            print(f'save file in {ext}')
            write_graph_in_file(G_new, graph_name, ext, folder_no) 
            cond = nx.conductance(G_new, my_dict['query'])
            query_nodes = my_dict['query']
            print(f'query nodes are {query_nodes}')
            print(f'******** Condactance is equal to {cond}\n\n')
            #nx.draw_networkx(G_new)
            #plt.show()
        print(my_dict)
        
def make_it_clique(G, query_nodes):
    G_new = G.copy()
    for q_node1 in query_nodes:
        for q_node2 in query_nodes:
            if q_node1 != q_node2:
                if(G.has_edge(q_node1, q_node2) == False): 
                    #print("Adding edge {q_node1} - {q_node2}")
                    G_new.add_edge(q_node1, q_node2)
    return G_new
                    
                


if __name__ == '__main__':
    graph_names = ['football', 'highschool', 'bio-celegans']
    #graph_names = ['football']
    for folder_no in range(10):
        for graph_name in graph_names:
            print(f'Graph name is  {graph_name} and no is {folder_no+1}')
            perc = [0.05, 0.1, 0.2]
            clcr = [[1],[i/10.0 for i in range(11)]]
            #clcr = [[(i+5)/10.0 for i in range(6)], [1]]
            make_data(graph_name, perc = perc, clcr = clcr, folder_no = folder_no+1, use_clique = True)
    sys.exit()
    
