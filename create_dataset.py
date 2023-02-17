#create the real and the synthetic graphs used for the experiments 
def make_random_renyi_graph(n = 200, p = 0.5):
    import networkx as nx
    G = nx.erdos_renyi_graph(n, p)
    with open(f'ssl_graphs/renyi_{n}_{int(p*100)}.txt', 'w') as f:  
        for edge in G.edges():
            print(edge[0], edge[1])
            if edge[0] ==  edge[1]: continue
            f.write(str(edge[0])+' '+str(edge[1])+'\n')


def make_barabasi_albert(n = 200, m = 50):
    import networkx as nx
    G = nx.barabasi_albert_graph(n, m)
    with open(f'ssl_graphs/barabasi_{n}_{m}.txt', 'w') as f:  
        for edge in G.edges():
            print(edge[0], edge[1])
            if edge[0] ==  edge[1]: continue
            f.write(str(edge[0])+' '+str(edge[1])+'\n')

def powerlaw_cluster(n = 200, m = 50, p = 0.5):
    import networkx as nx
    G = nx.powerlaw_cluster_graph(n, m, p, seed=None)
    with open(f'ssl_graphs/powerlaw_{n}_{m}_{int(p*100)}.txt', 'w') as f:  
        for edge in G.edges():
            print(edge[0], edge[1])
            if edge[0] ==  edge[1]: continue
            f.write(str(edge[0])+' '+str(edge[1])+'\n')

def create_ant():
    import numpy as np
    edge_list = np.loadtxt(f'ssl_graphs/ant.txt', usecols=range(2))
    with open(f'ssl_graphs/ant.txt', 'w') as f:  
        for edge in edge_list:
            i = edge[0]
            j = edge[1]
            if i == j: continue
            print(i, j)
            f.write(str(int(i))+' '+str(int(j))+'\n')

if __name__ == '__main__':
   make_random_renyi_graph()
   make_barabasi_albert()
   powerlaw_cluster()
   create_ant()
