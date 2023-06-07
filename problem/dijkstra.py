import torch
import networkx as nx
import numpy as np

class DijkstraSolution:
    def __init__(self, original_A, votes, experiments_to_make, weight_variant, threshold_percentage, majority_variant, length_of_query):
        self.original_A = original_A
        self.votes = votes
        self.experiments_to_make = experiments_to_make
        self.variant = weight_variant
        self.threshold_percentage = threshold_percentage
        self.length_of_query = length_of_query
        self.majority_variant = majority_variant
        self.inf = self.__get_infinite_length()

    def __get_infinite_length(self):
        if self.variant == "linear":
            return self.length_of_query * self.experiments_to_make + 1
        if self.variant == "quadratic":
            return self.length_of_query * self.experiments_to_make**2 + 1
        if self.variant == "cubic":
            return self.length_of_query * self.experiments_to_make**3 + 1
        if self.variant == "quartic":
            return self.length_of_query * self.experiments_to_make**4 + 1
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
        if self.variant == "quartic":
            return vote ** 4
        raise Exception("The variant was not recognized!")

    def __update_dijkstra_votes(self, dijkstra_votes, dijkstra_result, source_votes):
        longest_approved_distance = None
        for counter, (idx, distance) in enumerate(dijkstra_result.items()):
            # Multiple nodes have the same length, therefore the threshold is extended
            is_extra_node_approved = distance == longest_approved_distance and distance < self.inf
            if counter >= self.length_of_query and not is_extra_node_approved:
                return
            
            # Exactly |Vq| nodes have been collected. Only if any other node have same distance, it will be approved
            if counter == self.length_of_query - 1:
                longest_approved_distance = distance

            dijkstra_votes[idx] += self.__solution_weight(source_votes)

    def __solution_weight(self, source_votes):
        if self.majority_variant == "constant":
            return 1
        if self.majority_variant == "linear":
            return 1 / (1 + source_votes)
        raise Exception("Majority variant was not recognized!")

    def __find_solution_majority(self, dijkstra_votes, no_of_sources, most_source_votes):
        weight = self.__solution_weight(most_source_votes)
        solution_threshold = int(weight * no_of_sources * self.threshold_percentage)
        v_list = [0 if vote > solution_threshold else 1 for vote in dijkstra_votes]

        return torch.tensor(v_list)

    def __find_sources(self, votes):
        _threshold_percentage = 1 - self.threshold_percentage
        source_threshold = int(self.experiments_to_make * _threshold_percentage)
        sources = [idx for idx in range(len(votes)) if votes[idx] <= source_threshold]
        return sources

    def solution(self):
        G = nx.from_numpy_matrix(self.original_A.clone().detach().numpy())
        _votes = self.votes.clone().detach().numpy()

        def weight_function(u, v, direction):
            vote = _votes[v]
            weight = self.__get_weight(vote)
            return weight

        # If majority of the experiments agree on a node, include it as a source for Dijkstra.
        sources = self.__find_sources(_votes)

        print("length of sources:", len(sources))
        print("sources:", sources)

        # Run dijkstra for each source and save votes
        dijkstra_votes = np.zeros_like(_votes)
        for source in sources:
            dijkstra_result = nx.single_source_dijkstra_path_length(G, source, weight=weight_function)
            self.__update_dijkstra_votes(dijkstra_votes, dijkstra_result, _votes[source])

        # If majority of the sources agree on a node, include it in solution
        v = self.__find_solution_majority(dijkstra_votes, len(sources), min(_votes))

        return v
