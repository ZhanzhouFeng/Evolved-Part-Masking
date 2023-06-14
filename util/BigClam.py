import torch
import torch.nn as nn
import itertools
import math

class GraphLogLikelihood(nn.Module):
    """
    LogLikelihood log(P(G|F)) that a given community membership table F generates the ground truth graph G.
    """

    # def forward(self, input: torch.Tensor) -> torch.Tensor:
    #     """
    #     Computes the log likelihood log(P(G|F)).
    #     :param input: A tensor representing the community membership table F of shape (#nodes, #communities).
    #     :return: scalar tensor.
    #     """
    #
    #     loss = torch.sum(torch.log(1 - torch.exp((-1) * torch.einsum('ij,ij->i',
    #                                                                  input[self.edge_index[0]],
    #                                                                  input[self.edge_index[1]])))) \
    #            - torch.sum(torch.einsum('ij,ij->i', input[self.non_edge_index[0]], input[self.non_edge_index[1]]))
    #     return loss

    def forward(self,F,A):
        F=F/torch.norm(F, p=2, dim=-1,keepdim=True)
        A_soft =torch.bmm(F,F.permute(0,2,1))
        # A_soft=torch.mul(A,A_soft)

        FIRST_PART = A * torch.log(torch.exp(torch.tensor(1))- torch.exp(-1. * A_soft))
        sum_edges = torch.sum(FIRST_PART)

        SECOND_PART = (1 - A) * A_soft
        sum_nedges = torch.sum(SECOND_PART)

        log_likeli = sum_edges - sum_nedges
        return log_likeli


class BigCLAM:
    """
    Implementation of BigCLAM described in the paper
    "Overlapping Community Detection at Scale: A Nonnegative Matrix Factorization Approach"
    (https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf)
    """

    def __init__(self, convergence_threshold: float = 0.8):
        """
        :param convergence_threshold: A scalar threshold for when the training should stop.
        """

        self.convergence_threshold = convergence_threshold
        self.objective = GraphLogLikelihood()


    def run(self, adj_m: torch.Tensor, n_communities: int,
            device: torch.device, lr: float = 1.0e-3) -> (torch.Tensor, float):
        """
        Computes the community membership table F for a given graph.

        :param edge_index: A tensor that contains the edges / defines the graph structure.
        :param n_nodes: Number of nodes in the graph.
        :param n_communities: Number of possible communities.
        :param device: device to run the computations on.
        :param lr: learning rate for community membership table training.
        :return: the trained community membership table F and the corresponding log likelihood.
        """

        N,n_nodes,_=adj_m.shape

        membership = nn.Parameter(torch.empty((N,n_nodes, n_communities), device=device))
        torch.nn.init.kaiming_normal_(membership)
        optim = torch.optim.SGD(params=[membership], lr=lr)

        # Training procedure.
        likelihood_value = 0
        # while likelihood_value < self.convergence_threshold:
        for i in range(5):
            likelihood = self.objective(membership,adj_m)
            likelihood_value = likelihood.item()
            optim.zero_grad()
            likelihood.backward()
            optim.step()
            # print(likelihood_value)

        return membership.detach(), likelihood_value
