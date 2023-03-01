#generates graphs for partial matching of specified type of graph, size of part and rest, number of connected edges
import numpy as np
import networkx as nx
from datetime import datetime
import os
import argparse

def parse_args():
    parser=argparse.ArgumentParser(description= "generate graphs")
    parser.add_argument('--graph', nargs='?', default='ca-netscience')
    parser.add_argument('--sizes_part', type=int, default= [20,30,40,50], help='sizes of parts')
    return parser.parse_args()

def main(args):
    now = datetime.now()
    folder=now.strftime("%d-%m-%Y_%H-%M-%S")
    folder=args.graph+'_'+folder
    folder = './Data'
    # TODO uncomment if you want to generate these again
    # generate_ssl_graph(folder)
    # generate_ssl_graph_perc(folder)

def generate_ssl_graph(folder):

    graph_names = ['ant', 'football', 'highschool', 'malaria', 'powerlaw_200_50_50', 'renyi_200_50', 'barabasi_200_50']
    part_sizes = [0.1,0.2,0.3]
    its=11
    for i in range(len(graph_names)):
        for it in range(its):
            print(i)
            graph = graph_names[i]
            for ps in part_sizes:
                is_con = False
                while is_con == False:
                    G = nx.read_edgelist('ssl_graphs/'+ graph + '.txt', data=False)
                    G = nx.relabel.convert_node_labels_to_integers(G, 0)
                    # nx.write_edgelist(G,'arenas.txt')
                    # print(G.nodes())
                    graph_size = len(G.nodes())
                    size_part = round(graph_size * ps)
                    print(size_part)
                    print(graph_size / 2)

                    # print(graph_size)
                    start_node = np.random.randint(1, graph_size)
                    neighbor_list = []
                    # choose part nodes
                    list_part = [start_node]
                    neigh_probs = []
                    for n in G.neighbors(start_node):
                        neighbor_list.append(n)
                        neigh_probs.append(1.0)
                    while (len(list_part) < size_part and len(neighbor_list) > 0):
                        p_arr = np.asarray(neigh_probs)

                        p_norm = p_arr / np.sum(p_arr)
                        # print(len(neighbor_list))
                        new_node = np.random.choice(neighbor_list, p=p_norm)
                        new_node_idx = neighbor_list.index(new_node)
                        list_part.append(new_node)
                        neighbor_list.remove(new_node)
                        neigh_probs.pop(new_node_idx)

                        for nn in G.neighbors(new_node):
                            if (nn not in list_part):
                                if (nn not in neighbor_list):
                                    neighbor_list.append(nn)
                                    neigh_probs.append(1.0)
                                else:
                                    cur_idx = neighbor_list.index(nn)
                                    neigh_probs[cur_idx] = neigh_probs[cur_idx] + 1.0

                            # for nnn in neighbor_list:
                            #    if(G.degree(nnn)==1):
                            #       list_part.append(nnn)

                            # collect components
                    rest_nodes = [i for i in range(G.number_of_nodes()) if i not in list_part]
                    is_con = nx.is_connected(nx.subgraph(G, list_part)) and nx.is_connected(nx.subgraph(G, rest_nodes))
                    if(is_con==False): continue
                    print(f'is connected -> {is_con}')
                    G_del = nx.read_edgelist('ssl_graphs/' + graph + '.txt')
                    G_del = nx.relabel.convert_node_labels_to_integers(G_del, 0)
                    G_del.remove_nodes_from(list_part)

                    l = [c for c in sorted(nx.connected_components(G_del), key=len, reverse=True)]

                    for ck in range(1, len(l)):
                        cur_comp = l[ck]
                        for q in cur_comp:
                            list_part.append(q)
                    
                         

                len_part = len(list_part)
                print('************** len part is {len_part}')
                G_sub = G.subgraph(list_part)
                print("size of subgraph: %d" % (len(G_sub.nodes())))

                if (True):
                    directo = folder + '/' + graph+'/'+str(int(ps*100))+'/'+ str(it)
                    if not os.path.exists(directo):
                        os.makedirs(directo)
                    #nx.write_edgelist(G_sub, folder + '/' + graph+'/'+str(int(ps*100))+'/' + graph + '_' + str(int(ps*100))+'_'+str(it) + '_part.txt')
                    part_node_arr = np.array(list_part)
                    np.savetxt(folder + '/' + graph+ '/' +str(int(ps*100))+'/' + str(it) +'/' + graph + '_' + str(int(ps*100))+'_nodes.txt',
                               part_node_arr, fmt='%d')
                    directo = folder + '/' + graph+'/'+str(int(ps*100))
                    nx.write_edgelist(G, folder + '/' + graph+ '/'+str(int(ps*100))+'/'+ str(it) +'/' + graph + '_' + str(int(ps*100)) + '.txt')

def edgelist_to_adjmatrix(edgeList_file):
    edge_list = np.loadtxt(edgeList_file, usecols=range(2))
    print('edgelist = ', edge_list)
    n = int(np.amax(edge_list) + 1)
    # n = int(np.amax(edge_list))

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
    import random
    return random.randint(0,max_num-1)

def random_numbers(max_num, total_el):
    my_list = []
    while len(my_list) < total_el:
        rn = random_number(max_num)
        if rn in my_list:
            continue
        my_list.append(rn)
    return my_list

def generate_ssl_graph_perc(folder):
    #insecta-ant-colony6-day09
    graph_names = ['ant', 'football', 'highschool', 'malaria', 'powerlaw_200_50_50', 'renyi_200_50', 'barabasi_200_50']
    part_sizes = [0.1,0.2,0.3]
    perc = [i/10.0 for i in range(11)]
    its=11
    
    for i in range(len(graph_names)):
        print()
        for it in range(its):
            print()
            graph = graph_names[i]
            for ps in part_sizes:
                edge_file = folder + '/' + graph+ '/'+str(int(ps*100))+'/'+ str(it) +'/' + graph + '_' + str(int(ps*100)) + '.txt'
                print(edge_file)
                #G = nx.read_edgelist(edge_file, data=False)
                #G = nx.relabel.convert_node_labels_to_integers(G, 0)
                A1=edgelist_to_adjmatrix(edge_file)

                G=nx.from_numpy_matrix(A1)
                print(f'Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges')
                part_file = folder + '/' + graph+ '/'+str(int(ps*100))+'/'+ str(it) +'/' + graph + '_' + str(int(ps*100)) + '_nodes.txt'
                query_nodes = np.loadtxt(part_file)
                is_con = nx.is_connected(nx.subgraph(G, query_nodes))
                print(f'The query nodes are connected = {is_con}')
                
                nodes_from_q_to_q = nx.subgraph(G, query_nodes).number_of_edges()
                rest_nodes = [i for i in range(G.number_of_nodes()) if i not in query_nodes]
                nodes_from_rest_to_rest = nx.subgraph(G, rest_nodes).number_of_edges()
                nodes_from_q_to_rest = G.number_of_edges() - nodes_from_q_to_q - nodes_from_rest_to_rest
                
                for per in perc:
                    Gnew = G.copy()
                    counter = nodes_from_q_to_rest
                    new_edges = int(per * nodes_from_q_to_rest)
                    rem_edges = random_numbers(nodes_from_q_to_rest, counter - new_edges)
                    counter_in = 0
                    for qnode in query_nodes:
                        for restnode in rest_nodes:
                            if(Gnew.has_edge(qnode, restnode)):
                                if counter_in in rem_edges:
                                    Gnew.remove_edge(qnode, restnode)
                                    counter -= 1
                                counter_in += 1
                    print(f'counter = {counter} and new edges are {new_edges}')
                    new_edge_file = folder + '/' + graph+ '/'+str(int(ps*100))+'/'+ str(it) +'/' + graph + '_' + str(int(ps*100)) +'_'+ str(int(per*100))+'.txt'
                    print(f'New file looks like {new_edge_file}')
                    nx.write_edgelist(Gnew, new_edge_file)
                            
if __name__ == '__main__':
    args = parse_args()
    main(args)
