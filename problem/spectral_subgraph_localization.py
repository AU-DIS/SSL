from networkx.algorithms.centrality.betweenness_subset import dijkstra
import torch
import numpy as np
from numpy import histogram, random, random
from torch import tensor
from tqdm import tqdm
import matplotlib.pyplot as plt
from optimization.algs.prox_grad import PGM
from optimization.prox.prox import ProxL21ForSymmetricCenteredMatrix, l21, \
    ProxL21ForSymmCentdMatrixAndInequality, ProxSymmetricCenteredMatrix, ProxId, \
    ProxNonNeg
from tests.optimization.util import double_centering, set_diag_zero
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.vq import kmeans2
from skimage.filters.thresholding import threshold_otsu
import networkx as nx
import kmeans1d
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from time import sleep
from scipy.linalg import qr, pinv
from sklearn.cluster import SpectralClustering
from enum import Enum
from problem.dijkstra import DijkstraSolution

def block_stochastic_graph(n1, n2, p_parts=0.7, p_off=0.1):
    n = n1 + n2
    p11 = set_diag_zero(p_parts * torch.ones(n1, n1))

    p22 = set_diag_zero(p_parts * torch.ones(n2, n2))

    p12 = p_off * torch.ones(n1, n2)

    p = torch.zeros([n, n])
    p[0:n1, 0:n1] = p11
    p[0:n1, n1:n] = p12
    p[n1:n, n1:n] = p22
    p[n1:n, 0:n1] = p12.T

    return p

from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support
def balanced_acc(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)

def recall(y_true, y_pred):
    return recall_score(y_true, y_pred)

def precision(y_true, y_pred):
    return precision_score(y_true, y_pred)

def f1(y_true, y_pred):
    return f1_score(y_true, y_pred)

def count_nodes(v_binary):
    return len(v_binary) - np.count_nonzero(v_binary)

def prec_recall_fscore(y_true, y_pred):
    prec, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred)
    return prec, recall, fscore

def rate_solution(v_gt, v_binary):
    print("Balanced accuracy:", balanced_acc(v_gt, v_binary))

    prec, recall, fscore = prec_recall_fscore(v_gt, v_binary)
    print("Precision:", prec)
    print("Recall:", recall)
    print("F-score:", fscore)

    print("Nodes in solution:", count_nodes(v_binary))
    print()

class Solution_algo(Enum):
    THRESHOLD = 1
    DIJKSTRA = 2

class VotingSubgraphIsomorpishmSolver:
    def __init__(self, A, ref_spectrum, problem_params, solver_params, v_gt, query, save_loss_terms=True, experiments_to_make = 20, edge_removal = 0.3):
        self.A = A
        self.ref_spectrum = ref_spectrum
        self.problem_params = problem_params
        self.solver_params = solver_params
        self.save_loss_terms = save_loss_terms
        self.v_gt = v_gt
        self.query = query
        self.length_of_query = len(query)
        self.experiments_to_make = experiments_to_make
        self.edge_removal = edge_removal

    def solve(self, max_outer_iters=10, max_inner_iters=10, show_iter=10, verbose=True):
        original_A = self.A.detach().clone()
        edge_list = adjmatrix_to_edgelist(self.A)
        print("edge_removal:", self.edge_removal)
        edges_removal_array = [self.edge_removal] * self.experiments_to_make # FAKE IT
 
        n = original_A.shape[0]
        votes = torch.zeros(n)

        for i in range(self.experiments_to_make):
            print(i)

            # remove edges
            no_of_edges_to_remove = int(len(edge_list) * edges_removal_array[i])
            edges_to_remove = find_random_edges(edge_list, no_of_edges_to_remove)
            modified_A = remove_edges(original_A.detach().clone(), edges_to_remove)

            # solving ssl for the modified graph
            solver = \
                SubgraphIsomorphismSolver(modified_A, self.ref_spectrum, self.problem_params, self.solver_params)
            v, E = \
                solver.solve(max_outer_iters, max_inner_iters, show_iter, verbose)

            # adding votes
            v_binary, _ = solver.threshold(v_np=v.detach().numpy())
            votes += v_binary

        # v, E = VotingSubgraphIsomorpishmSolver.find_solution(original_A, votes, experiments_to_make, algo=Solution_algo.DIJKSTRA, threshold_percentage=0.3)
        
        # Pretty printing solutinos for experimenting purposes
        # print_solutions(original_A, votes, experiments_to_make, self.v_gt)

        return votes

    def find_solution(self, original_A, votes, experiments_to_make, algo=Solution_algo.THRESHOLD, threshold=0.2, threshold_percentage=0.5, dijkstra_majority_variant="constant"):
        v = None

        if algo == Solution_algo.THRESHOLD:
            v = find_voting_majority(votes, experiments_to_make, threshold)
        elif algo == Solution_algo.DIJKSTRA:
            dijkstra = DijkstraSolution(original_A, votes, experiments_to_make, "cubic", threshold_percentage, dijkstra_majority_variant, self.length_of_query)
            v = dijkstra.solution()

        if v is None:
            raise Exception("Should have found the final vertices for the solution, but did not find any!")

        E = SubgraphIsomorphismSolver.E_from_v(v.detach(), original_A)
        return v, E

class SubgraphIsomorphismSolver:

    def __init__(self, A, ref_spectrum, problem_params, solver_params,
                 save_loss_terms=True):

        """
        Proximal algorithm solver for subgraph spectral matching.
        :param L: Laplacian of full graph
        :param ref_spectrum: spectrum of reference graph (i.e., the subgraph)
        :param params: parameters for the algorithm and solver. For example:
            params =
            {'maxiter': 100,
              'show_iter': 10,
              'mu_spectral': 1,
              'mu_l21': 1,
              'mu_MS': 1,
              'mu_split': 1,
              'mu_trace': 0.0,
              'lr': 0.02,
              'v_prox': ProxNonNeg(),
              'E_prox': ProxL21ForSymmCentdMatrixAndInequality(solver="cvx", L=L,
                                                                trace_upper_bound=
                                                                1.1*torch.trace(L)),
              'trace_val': 0
              }
        :param save_loss_terms: flag for saving the individual loss terms
        """

        self.A = A
        self.D = torch.diag(A.sum(dim=1))
        self.L = lap_from_adj(A)
        self.ref_spectrum = ref_spectrum
        self.subgraph_size = ref_spectrum.shape[0]
        self.spectrum_alignment_terms = []
        self.MS_reg_terms = []
        self.trace_reg_terms = []
        self.graph_split_terms = []
        self.l21_terms = []
        self.loss_vals = []
        self.save_individual_losses = save_loss_terms
        self.solver_params = solver_params
        self.problem_params = problem_params
        self.set_problem_params(problem_params)
        self.set_solver_params(solver_params)

        # init
        self.set_init()

    def set_init(self, v0=None, E0=None):
        n = self.L.shape[0]
        if v0 is None:
            eig_max = torch.max(self.ref_spectrum)
            c = np.sqrt(self.A.shape[0] - self.ref_spectrum.shape[0]) * eig_max
            #c = 2*np.sqrt(self.A.shape[0] - self.ref_spectrum.shape[0]) * eig_max
            v0 = (c / np.sqrt(n)) * np.ones(n)
            #v0 = (c / (n)) * np.ones(n)
            self.v = torch.tensor(v0, requires_grad=self.train_v, dtype=torch.float64)
        else:
            self.v = v0

        if E0 is None:
            # init
            E = torch.zeros([n, n], dtype=torch.float64)
            self.E = \
                double_centering(0.5 * (E + E.T)).requires_grad_(
                    requires_grad=self.train_E)
        else:
            self.E = E0

        self.E.requires_grad = self.train_E
        self.v.requires_grad = self.train_v

    def set_optim(self, v, E):
        lamb = self.mu_l21 * self.lr  # This setting is important!

        if self.train_E and self.train_v:
            optim_params = [{'params': v}, {'params': E}]
            proxs = [self.v_prox, self.E_prox]
        elif not self.train_E and self.train_v:
            optim_params = [{'params': v}]
            proxs = [self.v_prox]
        elif self.train_E and not self.train_v:
            optim_params = [{'params': E}]
            proxs = [self.E_prox]

        v.requires_grad = self.train_v
        E.requires_grad = self.train_E
        pgm = PGM(params=optim_params,
                  proxs=proxs,
                  lr=self.lr)
        return pgm, lamb

    def set_problem_params(self, problem_params):
        self.mu_spectral = problem_params['mu_spectral']
        self.mu_l21 = problem_params['mu_l21']
        self.mu_MS = problem_params['mu_MS']
        self.mu_trace = problem_params['mu_trace']
        self.mu_split = problem_params['mu_split']
        self.trace_val = problem_params['trace_val']
        self.weighted_flag = problem_params['weighted_flag']

    def set_solver_params(self, solver_params):
        self.v_prox = solver_params['v_prox']
        self.E_prox = solver_params['E_prox']
        self.lr = solver_params['lr']
        self.train_v = solver_params['train_v']
        self.train_E = solver_params['train_E']
        self.a_tol = solver_params['a_tol']
        self.r_tol = solver_params['r_tol']
        self.threshold_algo = solver_params['threshold_algo']

    def solve(self,
              max_outer_iters=10,
              max_inner_iters=10,
              show_iter=10,
              verbose=False):
        outer_iter_counter = 0
        converged_outer = False
        while not converged_outer:
            v, _ = self._solve(maxiter_inner=max_inner_iters,
                               show_iter=show_iter,
                               verbose=verbose)
            E, _ = self.E_from_v(self.v.detach(), self.A)
            self.set_init(E0=E, v0=v)
            self.v = v
            self.E = E
            outer_iter_counter += 1
            converged_outer = self._check_convergence(self.v.detach(), self.a_tol)
            converged_outer = converged_outer or (outer_iter_counter >= max_outer_iters)

        # Return v_binary and e_binary instead of v and E!
        return v, E

    def randomized_solve(self,
              max_outer_iters=10,
              max_inner_iters=10,
              show_iter=10,
              verbose=True):
        print("Please don't go here")
        original_A = self.A.detach().clone()
        edge_list = adjmatrix_to_edgelist(self.A)
        edges_removal_array = [0.005, 0.02]
        experiments_to_make = 30 # FAKE IT
        nodes_in_result = 31 # FAKE IT
 
        n = original_A.shape[0]
        votes = torch.zeros(n)

        for i in range(experiments_to_make):
            outer_iter_counter = 0
            converged_outer = False

            # remove edges
            no_of_edges_to_remove = int(len(edge_list) * edges_removal_array[i])
            edges_to_remove = find_random_edges(edge_list, no_of_edges_to_remove)
            modified_A = remove_edges(original_A.detach().clone(), edges_to_remove)
            self.A = modified_A
            self.L = lap_from_adj(modified_A)

            # solving ssl for the modified graph
            while not converged_outer:
                v, _ = self._solve(maxiter_inner=max_inner_iters,
                                   show_iter=show_iter,
                                   verbose=verbose)
                E, _ = self.E_from_v(self.v.detach(), self.A)
                self.set_init(E0=E, v0=v)
                self.v = v
                self.E = E
                outer_iter_counter += 1
                converged_outer = self._check_convergence(self.v.detach(), self.a_tol)
                converged_outer = converged_outer or (outer_iter_counter >= max_outer_iters)

            # adding votes
            v_binary, _ = self.threshold(v_np=v.detach().numpy())
            votes += v_binary

        # Finding the voting majority
        v = find_voting_majority(votes, experiments_to_make)
        E = self.E_from_v(v.detach(), original_A)

        # Return v_binary and e_binary instead of v and E!

        # Return v_binary and e_binary instead of v and E!
        return v, E

    def randomized_solve(self,
              max_outer_iters=10,
              max_inner_iters=10,
              show_iter=10,
              verbose=True):
        print("Please don't go here")
        original_A = self.A.detach().clone()
        edge_list = adjmatrix_to_edgelist(self.A)
        edges_removal_array = [0.005, 0.02] 
        experiments_to_make = 30 # FAKE IT
        nodes_in_result = 31 # FAKE IT
 
        n = original_A.shape[0]
        votes = torch.zeros(n)

        for i in range(experiments_to_make):
            outer_iter_counter = 0
            converged_outer = False

            # remove edges
            no_of_edges_to_remove = int(len(edge_list) * edges_removal_array[i])
            edges_to_remove = find_random_edges(edge_list, no_of_edges_to_remove)
            print("amount of bridges",nx.bridges(nx.from_edgeList(edge_list)))
            modified_A = remove_edges(original_A.detach().clone(), edges_to_remove)
            self.A = modified_A
            self.L = lap_from_adj(modified_A)

            # solving ssl for the modified graph
            while not converged_outer:
                v, _ = self._solve(maxiter_inner=max_inner_iters,
                                   show_iter=show_iter,
                                   verbose=verbose)
                E, _ = self.E_from_v(self.v.detach(), self.A)
                self.set_init(E0=E, v0=v)
                self.v = v
                self.E = E
                outer_iter_counter += 1
                converged_outer = self._check_convergence(self.v.detach(), self.a_tol)
                converged_outer = converged_outer or (outer_iter_counter >= max_outer_iters)

            # adding votes
            v_binary, _ = self.threshold(v_np=v.detach().numpy())
            votes += v_binary

        # Finding the voting majority
        v = find_voting_majority(votes, experiments_to_make)
        E = self.E_from_v(v.detach(), original_A)

        # Return v_binary and e_binary instead of v and E!
        return v, E

    def _solve(self, maxiter_inner=100, show_iter=10, verbose=False):
        L = self.L

        # Q, R = qr(L.numpy())
        # self.P = torch.eye(L.shape[0]) - torch.tensor(Q @ Q.T)

        ref_spectrum = self.ref_spectrum
        n = L.shape[0]

        # init
        v = self.v
        E = self.E

        # for linear inverse problems this is the optimal setting for the step size
        # s = torch.linalg.svdvals(A)
        # lr = 1 / (1.1 * s[0] ** 2)

        pgm, lamb = self.set_optim(v, E)

        full_loss_function = lambda ref, L, E, v: \
            self.smooth_loss_function(ref, L, E, v, False) \
            + self.non_smooth_loss_function(E)

        loss_vals = []

        groups = {'loss': ['loss']}
        plotlosses = PlotLosses(groups=groups, outputs=[MatplotlibPlot()])
        # converged_inner = False
        iter_count = 0
        for i in tqdm(range(maxiter_inner), disable=True):
            v_prev = self.v.detach()
            pgm.zero_grad()
            loss = \
                self.smooth_loss_function(ref_spectrum, L, E, v,
                                          self.save_individual_losses)
            loss_vals.append(
                full_loss_function(ref_spectrum, L, E.detach(), v.detach()))
            loss.backward()
            pgm.step(lamb=lamb)
            if (iter_count + 1) % show_iter == 0:
                self.plot_loss(plotlosses, loss_vals[-1])
            iter_count += 1
            converged_inner = self._check_convergence(v_prev=v_prev,
                                                      v=self.v.detach(),
                                                      r_tol=self.r_tol,
                                                      a_tol=self.a_tol)
            converged_inner = converged_inner or (iter_count >= maxiter_inner)
            if converged_inner:
                # Yes it's bad practice, but otherwise the progress bar won't update
                break
        #print("done")
        L_edited = L + E.detach() + torch.diag(v.detach())
        spectrum = torch.linalg.eigvalsh(L_edited)
        k = ref_spectrum.shape[0]

        self.loss_vals = [*self.loss_vals, *loss_vals]
        self.E = E
        self.v = v
        self.spectrum = spectrum.detach().numpy()
        # self.plots()

        if verbose:
            print(f"v= {v}")
            print(f"E= {E}")
            print(f"lambda= {spectrum[0:k]}")
            print(f"lambda*= {ref_spectrum}")
            print(f"||lambda-lambda*|| = {torch.norm(spectrum[0:k] - ref_spectrum)}")
        return v, E

    def _check_convergence(self, v, a_tol, v_prev=None, r_tol=None):
        # Todo: change conditions to follow kkt
        v_binary, E_binary = self.threshold(v_np=v.detach().numpy(),
                                            threshold_algo=self.threshold_algo)
        eig_max = torch.max(self.ref_spectrum).numpy()
        c = np.sqrt(self.A.shape[0] - self.ref_spectrum.shape[0]) * eig_max
        loss = self.smooth_loss_function(self.ref_spectrum, self.L,
                                         E_binary.detach(),
                                         c * v_binary.detach(), False)
        condition1 = loss < a_tol
        if r_tol is not None:
            condition2 = (torch.norm(v - v_prev) / torch.norm(v)) < r_tol
        else:
            condition2 = False
        return condition1 or condition2

    def non_smooth_loss_function(self, E):
        l21_loss = l21(E)

        if self.save_individual_losses:
            self.l21_terms.append(l21_loss)

        return self.mu_l21 * l21(E)

    def smooth_loss_function(self, ref_spectrum, L, E, v, save_individual_losses=False):

        spectrum_alignment_term = self.spectrum_alignment_loss(ref_spectrum, L, E, v,
                                                               self.weighted_flag)
        MS_reg_term = self.MSreg(L, E, v)
        trace_reg_term = self.trace_reg(E, self.trace_val)
        graph_split_term = self.graph_split_loss(L, E)

        smooth_loss_term = \
            self.mu_spectral * spectrum_alignment_term \
            + self.mu_MS * MS_reg_term \
            + self.mu_trace * trace_reg_term \
            + self.mu_split * graph_split_term

        if save_individual_losses:
            self.spectrum_alignment_terms.append(
                spectrum_alignment_term.detach().numpy())
            self.MS_reg_terms.append(MS_reg_term.detach().numpy())
            self.trace_reg_terms.append(trace_reg_term.detach().numpy())
            self.graph_split_terms.append(graph_split_term.detach().numpy())

        return smooth_loss_term

    def spectrum_alignment_loss(self, ref_spectrum, L, E, v, weighted_flag=True):
        k = ref_spectrum.shape[0]
        Hamiltonian = L + E + torch.diag(v)
        spectrum = torch.linalg.eigvalsh(Hamiltonian)
        if weighted_flag:
            weights = torch.tensor([1 / w if w > 1e-8 else 1 for w in
                                    ref_spectrum])
        else:
            weights = torch.ones_like(ref_spectrum)
        loss = torch.norm((spectrum[0:k] - ref_spectrum) * weights) ** 2
        return loss

    @staticmethod
    def graph_split_loss(L, E):
        L_edited = L + E
        spectrum = torch.linalg.eigvalsh(L_edited)
        # loss = torch.norm(spectrum[0:2]) ** 2
        loss = spectrum[1]
        # print(loss)
        return loss

    @staticmethod
    def MSreg(L, E, v):
        return v.T @ (L + E) @ v

    @staticmethod
    def trace_reg(E, trace_val=0):
        return (torch.trace(E) - trace_val) ** 2

    @staticmethod
    def E_from_v(v, A):
        v_ = SubgraphIsomorphismSolver.indicator_from_v(v)
        S = -torch.abs(v_[:, None] - v_[:, None].T) * A
        # E = torch.diag(S.sum(axis=1)) - S
        E = lap_from_adj(S)
        return E, S

    @staticmethod
    def indicator_from_v(v):
        v_ = v - torch.min(v)
        if torch.max(v_) != 0:
            v_ = v_ / torch.max(v_)
        # v_ = torch.ones_like(v_,)-v_
        return v_

    @staticmethod
    def indicator_from_v_np(v_np):
        v_ = v_np - np.min(v_np)
        if np.max(v_) != 0:
            v_ = v_ / np.max(v_)
        # v_ = torch.ones_like(v_,)-v_
        return v_

    def plot_loss(self, plotlosses, loss_val, sleep_time=.00001):
        plotlosses.update({
            'loss': loss_val,
        })
        plotlosses.send()
        sleep(sleep_time)

    def plot(self, plots):
        """
        produces various plots
        :param plots: flags for which plots to produce.
        The following plots are supported:
                plots = {'full_loss': True,
                        'E': True,
                        'v': True,
                        'diag(v)': True,
                        'v_otsu': True,
                        'v_kmeans': True,
                        'A edited': True,
                        'L+E': True,
                        'ref spect vs spect': True,
                        'individual loss terms': True}
        """
        E = self.E.detach().numpy()
        v = self.v.detach().numpy()

        if plots['full_loss']:
            plt.plot(self.loss_vals, 'b')
            plt.title('full loss')
            plt.xlabel('iter')
            plt.show()

        if plots['E']:
            ax = plt.subplot()
            im = ax.imshow(E - np.diag(np.diag(E)))
            divider = make_axes_locatable(ax)
            ax.set_title('E -diag(E)')
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()

        if plots['L+E']:
            ax = plt.subplot()
            L_edited = E + self.L.numpy()
            im = ax.imshow(L_edited)
            divider = make_axes_locatable(ax)
            ax.set_title('L+E')
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()

        if plots['A edited']:
            ax = plt.subplot()
            A_edited = -set_diag_zero(E + self.L.numpy())
            im = ax.imshow(A_edited)
            divider = make_axes_locatable(ax)
            ax.set_title('A edited')
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()

        if plots['v']:
            plt.plot(np.sort(v), 'xr')
            plt.title('v')
            plt.show()

        if plots['diag(v)']:
            ax = plt.subplot()
            im = ax.imshow(np.diag(v))
            divider = make_axes_locatable(ax)
            ax.set_title('diag(v)')
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()

        if plots['v_otsu']:
            ax = plt.subplot()
            # _, v_clustered = kmeans2(self.v, 2, minit='points')
            v = self.indicator_from_v_np(v)
            threshold = threshold_otsu(v, nbins=10)
            v_otsu = (v > threshold).astype(float)
            im = ax.imshow(np.diag(v_otsu))
            divider = make_axes_locatable(ax)
            ax.set_title('diag(v_otsu)')
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()

        if plots['v_kmeans']:
            ax = plt.subplot()
            # _, v_clustered = kmeans2(self.v, 2, minit='points')
            v = self.indicator_from_v_np(v)
            v_clustered, centroids = kmeans1d.cluster(v, k=2)
            im = ax.imshow(np.diag(v_clustered))
            divider = make_axes_locatable(ax)
            ax.set_title('diag(v_kmeans)')
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()

        if plots['ref spect vs spect']:
            k = self.ref_spectrum.shape[0]
            plt.plot(self.ref_spectrum.numpy(), 'og')
            plt.plot(self.spectrum[:k], 'xr')
            plt.title('ref spect vs first k eigs of subgraph')
            plt.show()

        if plots['individual loss terms']:
            font_color = "r"
            # Create two subplots and unpack the output array immediately
            fig, axes = plt.subplots(nrows=3, ncols=2)
            ax = axes.flat
            ax[0].loglog(self.spectrum_alignment_terms)
            ax[0].set_ylabel('loss', c=font_color)
            ax[0].set_xlabel('iteration', c=font_color)
            ax[0].set_title(f'spectral alignment, mu = {self.mu_spectral}',
                            c=font_color)

            ax[1].loglog(self.MS_reg_terms)
            ax[1].set_ylabel('loss', c=font_color)
            ax[1].set_xlabel('iteration', c=font_color)
            ax[1].set_title(f'MS reg, mu={self.mu_MS}', c=font_color)

            ax[2].loglog(self.graph_split_terms)
            ax[2].set_ylabel('loss', c=font_color)
            ax[2].set_xlabel('iteration', c=font_color)
            ax[2].set_title(f'graph split, mu={self.mu_split}', c=font_color)

            ax[3].loglog(self.loss_vals)
            ax[3].set_ylabel('loss', c=font_color)
            ax[3].set_xlabel('iteration', c=font_color)
            ax[3].set_title('full loss', c=font_color)

            ax[4].loglog(self.l21_terms)
            ax[4].set_ylabel('loss', c=font_color)
            ax[4].set_xlabel('iteration', c=font_color)
            ax[4].set_title(f'l21 reg, mu = {self.mu_l21}', c=font_color)

            ax[5].loglog(self.trace_reg_terms)
            ax[5].set_ylabel('loss', c=font_color)
            ax[5].set_xlabel('iteration', c=font_color)
            ax[5].set_title(f'trace reg, mu = {self.mu_trace}', c=font_color)

            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                wspace=0.5,
                                hspace=2)
            plt.show()

    @staticmethod
    def plot_on_graph(A, v, E, subset_nodes, pos=None):
        """
        plots the potentials E and v on the full graph
        :param A: adjacency of full graph
        :param n_subgraph: size of subgraph
        """
        if pos is None:
            pos = nx.spring_layout(nx.from_numpy_matrix(A))

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[10, 10])
        ax = axes.flat

        vmin = np.min(v)
        vmax = np.max(v)

        G = nx.from_numpy_matrix(A)
        # pos = nx.spring_layout(G)
        # pos = nx.spring_layout(G)
        # plt.rcParams["figure.figsize"] = (20,20)

        # for edge in G.edges():

        for u, w, d in G.edges(data=True):
            d['weight'] = E[u, w]

        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

        cmap = plt.cm.gnuplot
        # ax = plt.subplot()
        nx.draw(G, node_color=v, edgelist=edges, vmin=vmin, vmax=vmax, cmap=cmap,
                node_size=30,
                pos=pos, ax=ax[0])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        ax[0].set_title('Nodes colored by potential v')
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sm, ax=ax[0], cax=cax)
        ax[0].set_aspect('equal', 'box')

        #  plt.savefig(file+'.png')

        vmin = np.min(weights)
        vmax = np.max(weights)
        # subset_nodes = range(n_subgraph)
        # subset_nodes = np.loadtxt(data_path + graph_name + '_nodes.txt').astype(int)

        color_map = []
        for node in G:
            if node in subset_nodes:
                color_map.append('red')
            else:
                color_map.append('green')
        cmap = plt.cm.gnuplot
        # ax = plt.subplot()
        nx.draw(G, node_color=color_map, edgelist=edges, edge_color=weights, width=2.0,
                edge_cmap=cmap, vmin=vmin,
                vmax=vmax, cmap=cmap, node_size=30, pos=pos, ax=ax[1])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        ax[1].set_title('Edges colored by E')
        #  plt.savefig(file+'.png')

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sm, ax=ax[1], cax=cax)
        ax[1].set_aspect('equal', 'box')

        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
        #                     wspace=2,
        #                     hspace=None)

        plt.show()

    def threshold(self, v_np, threshold_algo='1dkmeans'):
        if threshold_algo == '1dkmeans':
            v_ = v_np - np.min(v_np)
            v_ = v_ / np.max(v_)
            v_clustered, centroids = kmeans1d.cluster(v_, k=2)

        elif threshold_algo == 'smallest':
            # just take the smallest subgraph_size entries in v_np
            subgraph_size = self.ref_spectrum.shape[0]
            v_clustered = np.zeros_like(v_np)
            ind = np.argsort(v_np)
            v_clustered[ind[subgraph_size:]] = 1

        elif threshold_algo == 'spectral':
            E , S = self.E_from_v(tensor(v_np), self.A)
            affinity_matrix = (self.A + S.detach())
            #affinity_matrix = lap_from_adj(affinity_matrix)
            clustering = \
                SpectralClustering(n_clusters=2, assign_labels='discretize',
                                   random_state=0, affinity='precomputed').fit(
                    affinity_matrix)
            v_clustered = np.ones_like(v_np) - clustering.labels_

        v_clustered = torch.tensor(v_clustered, dtype=torch.float64)
        E_clustered, _ = self.E_from_v(v_clustered, self.A)
        return v_clustered, E_clustered


def lap_from_adj(A):
    return torch.diag(A.sum(axis=1)) - A
    # D_sqrt_reciprocal = torch.diag(A.sum(axis=1) **-0.5)
    # return D_sqrt_reciprocal@A@D_sqrt_reciprocal


def find_random_edges(edge_list, no_of_edges_to_remove):
    _edge_list = edge_list.copy()
    edges_to_remove = []
    n = len(edge_list)
    for _ in range(no_of_edges_to_remove):
        idx = random.randint(n)
        edge_to_remove = _edge_list.pop(idx)
        edges_to_remove.append(edge_to_remove)
        n -= 1
    return edges_to_remove

def remove_edges(A, edges_to_remove):
    for i, j in edges_to_remove:
        if A[i, j] == 0 or A[j, i] == 0:
            raise Exception("Expected edge, but did not find any")
        A[i, j] = A[j, i] = 0

    return A

def find_voting_majority(votes, iterations_used, threshold):
    normalized_votes = votes / iterations_used

    _threshold = 1 - threshold

    # find indices of nodes with vote ratio <= _threshold
    indices_of_top_nodes = []
    for idx, vote_ratio in enumerate(normalized_votes):
        if vote_ratio < _threshold:
            indices_of_top_nodes.append(idx)

    voting_majority = torch.full_like(normalized_votes, 1, dtype=torch.double)

    for idx in indices_of_top_nodes:
        voting_majority[idx] = 0

    return voting_majority

def adjmatrix_to_edgelist(A):
    # assumes A is a quadratic matrix
    n, _ = A.shape

    res = []
    for i in range(n):
        for j in range(i+1): # assumes A is symmetric
            if A[i, j] == 0:
                continue
            # an edge from i to j exists in A
            res.append((i, j))

    return res


def find_random_edges(edge_list, no_of_edges_to_remove):
    _edge_list = edge_list.copy()
    edges_to_remove = []
    n = len(edge_list)
    for _ in range(no_of_edges_to_remove):
        idx = random.randint(n)
        edge_to_remove = _edge_list.pop(idx)
        edges_to_remove.append(edge_to_remove)
        n -= 1
    return edges_to_remove

def remove_edges(A, edges_to_remove):
    for i, j in edges_to_remove:
        if A[i, j] == 0 or A[j, i] == 0:
            raise Exception("Expected edge, but did not find any")
        A[i, j] = A[j, i] = 0

    return A

def find_voting_majority(votes, iterations_used, threshold):
    normalized_votes = votes / iterations_used

    _threshold = 1 - threshold

    # find indices of nodes with vote ratio <= _threshold
    indices_of_top_nodes = []
    for idx, vote_ratio in enumerate(normalized_votes):
        if vote_ratio < _threshold:
            indices_of_top_nodes.append(idx)

    voting_majority = torch.full_like(normalized_votes, 1, dtype=torch.double)

    for idx in indices_of_top_nodes:
        voting_majority[idx] = 0

    return voting_majority

def adjmatrix_to_edgelist(A):
    # assumes A is a quadratic matrix
    n, _ = A.shape

    res = []
    for i in range(n):
        for j in range(i+1): # assumes A is symmetric
            if A[i, j] == 0:
                continue
            # an edge from i to j exists in A
            res.append((i, j))

    return res


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

def print_solutions(original_A, votes, experiments_to_make, v_gt):
    v, _ = VotingSubgraphIsomorpishmSolver.find_solution(original_A, votes, experiments_to_make, algo=Solution_algo.THRESHOLD, threshold=0.1)
    print("0.1 Threshold solution:")
    rate_solution(v_gt, v)

    v, _ = VotingSubgraphIsomorpishmSolver.find_solution(original_A, votes, experiments_to_make, algo=Solution_algo.THRESHOLD)
    print("0.2 Threshold solution:")
    rate_solution(v_gt, v)

    v, _ = VotingSubgraphIsomorpishmSolver.find_solution(original_A, votes, experiments_to_make, algo=Solution_algo.THRESHOLD, threshold=0.3)
    print("0.3 Threshold solution:")
    rate_solution(v_gt, v)

    v, _  = VotingSubgraphIsomorpishmSolver.find_solution(original_A, votes, experiments_to_make, algo=Solution_algo.THRESHOLD, threshold=0.4)
    print("0.4 Threshold solution:")
    rate_solution(v_gt, v)

    v, _  = VotingSubgraphIsomorpishmSolver.find_solution(original_A, votes, experiments_to_make, algo=Solution_algo.THRESHOLD, threshold=0.5)
    print("0.5 Threshold solution:")
    rate_solution(v_gt, v)

    v, _ = VotingSubgraphIsomorpishmSolver.find_solution(original_A, votes, experiments_to_make, algo=Solution_algo.THRESHOLD, threshold=0.6)
    print("0.6 Threshold solution:")
    rate_solution(v_gt, v)

    print("Dijkstra solution 0.1 source threshold:")
    v, _ = VotingSubgraphIsomorpishmSolver.find_solution(original_A, votes, experiments_to_make, algo=Solution_algo.DIJKSTRA, threshold_percentage=0.1)
    rate_solution(v_gt, v)

    print("Dijkstra solution 0.2 source threshold:")
    v, _ = VotingSubgraphIsomorpishmSolver.find_solution(original_A, votes, experiments_to_make, algo=Solution_algo.DIJKSTRA, threshold_percentage=0.2)
    rate_solution(v_gt, v)

    print("Dijkstra solution 0.3 source threshold:")
    v, _ = VotingSubgraphIsomorpishmSolver.find_solution(original_A, votes, experiments_to_make, algo=Solution_algo.DIJKSTRA, threshold_percentage=0.3)
    rate_solution(v_gt, v)

    print("Dijkstra solution 0.4 source threshold:")
    v, _ = VotingSubgraphIsomorpishmSolver.find_solution(original_A, votes, experiments_to_make, algo=Solution_algo.DIJKSTRA, threshold_percentage=0.4)
    rate_solution(v_gt, v)

    print("Dijkstra solution 0.5 source threshold:")
    v, _ = VotingSubgraphIsomorpishmSolver.find_solution(original_A, votes, experiments_to_make, algo=Solution_algo.DIJKSTRA)
    rate_solution(v_gt, v)

    print("Dijkstra solution 0.6 source threshold:")
    v, _ = VotingSubgraphIsomorpishmSolver.find_solution(original_A, votes, experiments_to_make, algo=Solution_algo.DIJKSTRA, threshold_percentage=0.6)
    rate_solution(v_gt, v)

    print("WEIGHTED Dijkstra solution 0.1 source threshold:")
    v, _ = VotingSubgraphIsomorpishmSolver.find_solution(original_A, votes, experiments_to_make, algo=Solution_algo.DIJKSTRA, threshold_percentage=0.1, dijkstra_majority_variant="linear")
    rate_solution(v_gt, v)

    print("WEIGHTED Dijkstra solution 0.2 source threshold:")
    v, _ = VotingSubgraphIsomorpishmSolver.find_solution(original_A, votes, experiments_to_make, algo=Solution_algo.DIJKSTRA, threshold_percentage=0.2, dijkstra_majority_variant="linear")
    rate_solution(v_gt, v)

    print("WEIGHTED Dijkstra solution 0.3 source threshold:")
    v, _ = VotingSubgraphIsomorpishmSolver.find_solution(original_A, votes, experiments_to_make, algo=Solution_algo.DIJKSTRA, threshold_percentage=0.3, dijkstra_majority_variant="linear")
    rate_solution(v_gt, v)

    print("WEIGHTED Dijkstra solution 0.4 source threshold:")
    v, _ = VotingSubgraphIsomorpishmSolver.find_solution(original_A, votes, experiments_to_make, algo=Solution_algo.DIJKSTRA, threshold_percentage=0.4, dijkstra_majority_variant="linear")
    rate_solution(v_gt, v)

    print("WEIGHTED Dijkstra solution 0.5 source threshold:")
    v, _ = VotingSubgraphIsomorpishmSolver.find_solution(original_A, votes, experiments_to_make, algo=Solution_algo.DIJKSTRA, threshold_percentage=0.5, dijkstra_majority_variant="linear")
    rate_solution(v_gt, v)

    print("WEIGHTED Dijkstra solution 0.6 source threshold:")
    v, _ = VotingSubgraphIsomorpishmSolver.find_solution(original_A, votes, experiments_to_make, algo=Solution_algo.DIJKSTRA, threshold_percentage=0.6, dijkstra_majority_variant="linear")
    rate_solution(v_gt, v)

if __name__ == '__main__':
    torch.manual_seed(12)

    n1 = 5
    n2 = 15
    n = n1 + n2
    p = block_stochastic_graph(n1, n2, p_parts=0.7, p_off=0.2)

    A = torch.tril(torch.bernoulli(p)).double()
    A = (A + A.T)
    D = torch.diag(A.sum(dim=1))
    L = lap_from_adj(A)

    plt.imshow(A)
    plt.title('A')
    plt.show()

    A_sub = A[0:n1, 0:n1]
    D_sub = torch.diag(A_sub.sum(dim=1))
    L_sub = D_sub - A_sub
    ref_spectrum = torch.linalg.eigvalsh(L_sub)
    params = {'maxiter': 100,
              'show_iter': 10,
              'mu_spectral': 1,
              'mu_l21': 1,
              'mu_MS': 1,
              'mu_split': 1,
              'mu_trace': 0.0,
              'lr': 0.02,
              'v_prox': ProxNonNeg(),
              # 'E_prox': ProxL21ForSymmetricCenteredMatrix(solver="cvx"),
              'E_prox': ProxL21ForSymmCentdMatrixAndInequality(solver="cvx", L=L,
                                                               trace_upper_bound=
                                                               1.1 * torch.trace(L)),
              'trace_val': 0
              }
    plots = {
        'full_loss': True,
        'E': True,
        'v': True,
        'diag(v)': True,
        'v_otsu': False,
        'v_kmeans': True,
        'A edited': True,
        'L+E': False,
        'ref spect vs spect': True,
        'individual loss terms': True}
    subgraph_isomorphism_solver = \
        SubgraphIsomorphismSolver(L, ref_spectrum, params)
    v, E = subgraph_isomorphism_solver.solve()
    subgraph_isomorphism_solver.plot(plots)
    subgraph_isomorphism_solver.plot_on_graph(A.numpy().astype(int),
                                              n1,
                                              subgraph_isomorphism_solver.v,
                                              subgraph_isomorphism_solver.E)

