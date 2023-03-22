import torch
import networkx as nx
import numpy as np

class DijkstraSolution:
    def __init__(self, original_A, votes, experiments_to_make, variant, threshold_percentage):
        self.original_A = original_A
        self.votes = votes
        self.experiments_to_make = experiments_to_make
        self.variant = variant
        self.threshold_percentage = threshold_percentage
        self.length_of_query = 12 # TODO fake it, tilføjes senere som argument
        self.inf = self.__get_infinite_length()

    def __get_infinite_length(self):
        if self.variant == "linear":
            return self.length_of_query * self.experiments_to_make + 1
        if self.variant == "quadratic":
            return self.length_of_query * self.experiments_to_make**2 + 1
        if self.variant == "cubic":
            return self.length_of_query * self.experiments_to_make**3 + 1
        raise Exception("The variant was not recognized!")

    def __get_weight(self, vote):
        if vote == self.experiments_to_make:
            return self.inf
        if self.variant == "linear":
            return vote
        if self.variant == "quadratic":
            return vote ** 2
        if self.variant == "cubic":
            return vote ** 3
        raise Exception("The variant was not recognized!")

    def __update_dijkstra_votes(self, dijkstra_votes, dijkstra_result):
        longest_approved_distance = None
        for counter, (idx, distance) in enumerate(dijkstra_result.items()):
            # Multiple nodes have the same length, therefore the threshold is extended
            if counter >= self.length_of_query and distance != longest_approved_distance:
                return
            
            # Exactly |Vq| nodes have been collected. Only if any other node have same distance, it will be approved
            if counter == self.length_of_query - 1:
                longest_approved_distance = distance

            dijkstra_votes[idx] += 1

    def solution(self):
        G = nx.from_numpy_matrix(self.original_A.clone().detach().numpy())
        _votes = self.votes.clone().detach().numpy()

        def weight_function(u, v, direction):
            vote = _votes[v]
            weight = self.__get_weight(vote)
            return weight

        # If majority of the experiments agree on a node, include it as a source for Dijkstra.
        _threshold_percentage = 1 - self.threshold_percentage
        source_threshold = int(self.experiments_to_make * _threshold_percentage)
        sources = [idx for idx in range(len(_votes)) if _votes[idx] <= source_threshold]

        print("length of sources:", len(sources))
        print("sources:", sources)

        # Run dijkstra for each source and save votes
        dijkstra_votes = np.zeros_like(_votes)
        for source in sources:
            dijkstra_result = nx.single_source_dijkstra_path_length(G, source, weight=weight_function)
            self.__update_dijkstra_votes(dijkstra_votes, dijkstra_result)

        # If majority of the sources agree on a node, include it in solution
        # solution_threshold = len(sources)//2
        solution_threshold = int(len(sources) * self.threshold_percentage)
        v_list = [0 if vote > solution_threshold else 1 for vote in dijkstra_votes]
        v = torch.tensor(v_list)

        return v
