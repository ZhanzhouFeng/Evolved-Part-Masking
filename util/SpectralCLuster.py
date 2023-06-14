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

    def forward(self,F,A,D):
        # mask = (F == F.max(dim=2, keepdim=True)[0]).to(dtype=torch.float32)
        # mask_norm=1/mask.sum(dim=1)
        # # P=torch.bmm(F.permute(0,2,1),D)
        # # P=torch.bmm(P,F)
        # # P_reciprocal=1/(P+1-torch.eye(P.shape[-1],device=P.device).repeat(P.shape[0],1,1))
        #
        # L=D-A
        # O=F.permute(0,2,1)@L@F
        #
        # # loss=(O*P_reciprocal)*(torch.eye(O.shape[-1],device=O.device).repeat(O.shape[0],1,1))
        #
        # O_diag=(O)*(torch.eye(O.shape[-1],device=O.device).repeat(O.shape[0],1,1))
        # loss=O_diag.sum(dim=-1)*mask_norm
        #
        # # loss=(O)*(torch.eye(O.shape[-1],device=O.device).repeat(O.shape[0],1,1))
        # loss=loss.sum()+0.1*torch.abs(F.permute(0,2,1)@F-torch.eye(O.shape[-1],device=O.device).repeat(O.shape[0],1,1)).sum()
        # print(loss.sum(),torch.abs(F.permute(0,2,1)@F-torch.eye(O.shape[-1],device=O.device).repeat(O.shape[0],1,1)).sum())
        # return loss


        L = D - A
        O=F.permute(0,2,1)@L@F
        O=O*(torch.eye(O.shape[-1],device=O.device).repeat(O.shape[0],1,1))
        alpha=5
        loss=O.sum()+alpha*torch.abs(F.permute(0,2,1)@F-torch.eye(F.shape[-1],device=F.device).repeat(F.shape[0],1,1)).sum()
        # print(O.sum(),torch.abs(F.permute(0,2,1)@F-torch.eye(F.shape[-1],device=F.device).repeat(F.shape[0],1,1)).sum())
        return loss


        # L = D - A
        # alpha=0.1
        # loss=L-alpha*(F@F.permute(0,2,1))
        # return loss.sum()



        F=F/torch.norm(F, p=2, dim=-1,keepdim=True)
        A_soft =torch.bmm(F,F.permute(0,2,1))
        # A_soft=torch.mul(A,A_soft)

        FIRST_PART = A * torch.log(torch.exp(torch.tensor(1))- torch.exp(-1. * A_soft))
        sum_edges = torch.sum(FIRST_PART)

        SECOND_PART = (1 - A) * A_soft
        sum_nedges = torch.sum(SECOND_PART)

        log_likeli = sum_edges - sum_nedges
        return log_likeli


class SpecteralCluster:
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
            device: torch.device, lr: float =12.25e-2) -> (torch.Tensor, float):
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

        all_epoch=15

        lambda1 = lambda epoch: lr * (all_epoch-epoch+1)/all_epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda1)
        D=torch.bmm(adj_m,torch.ones(adj_m.shape,device=adj_m.device))*(torch.eye(adj_m.shape[-1],device=adj_m.device).repeat(adj_m.shape[0],1,1))

        # Training procedure.
        likelihood_value = 0
        last_likelihood=1e5
        # while likelihood_value < self.convergence_threshold:
        for i in range(all_epoch):
            likelihood = self.objective(membership,adj_m,D)
            likelihood_value = likelihood.item()
            # if likelihood_value>last_likelihood:
            #     break
            last_likelihood=likelihood_value
            optim.zero_grad()
            likelihood.backward()
            optim.step()
            scheduler.step()
            # print(i,likelihood_value)

        return membership.detach(), likelihood_value
