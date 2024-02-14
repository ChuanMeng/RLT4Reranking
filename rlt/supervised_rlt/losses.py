import torch
import torch.nn.functional as F
import math

class BiCutLoss(torch.nn.Module):
    def __init__(self, alpha: float=None, r: float=None):
        super(BiCutLoss, self).__init__()

        self.alpha = alpha #
        self.r = r # estimate r by the proportion of relevant documents in the training data

        """
        0 is "truncation" (false)，1 is "continue" (True)
        """

    def forward(self, output, data):
        # output [B, S, 2]
        # labels  [B, S]

        labels = data["rel"]
        r = torch.ones_like(output) # [B, S, 2] defaults to the device of input

        for i in range(labels.shape[0]): # B
            for j in range(labels.shape[1]): # S
                r[i][j] = torch.tensor([(1 - self.alpha) / self.r, 0]) if labels[i][j] == 1 else torch.tensor([0, self.alpha / (1 - self.r)])

        loss_matrix = output.mul(r) # [B, S, 2]

        return torch.sum(loss_matrix).div(output.shape[0])


class ChoppyLoss(torch.nn.Module):
    def __init__(self):
        super(ChoppyLoss, self).__init__()
    
    def forward(self, output, data):
        # output [B, S, 1]
        # labels  [B, S]

        r = data["label"] # [B, S]

        loss_matrix = output.squeeze().mul(r) # [B,S]

        return -torch.sum(loss_matrix).div(output.shape[0])


class AttnCutLoss(torch.nn.Module):
    def __init__(self, tau: float = None):
        super(AttnCutLoss, self).__init__()
        self.tau = tau

    def forward(self, output: torch.Tensor, data):
        # output [B, S, 1]
        # labels  [B, S]

        r = data["label"] # [B, S]

        q = torch.exp(r.div(self.tau))  # [B, S]
        # norm_factor [B,1]
        norm_factor = torch.sum(q, axis=1).unsqueeze(dim=1)
        # [B, S]
        q = q.div(norm_factor)

        # [B, S]
        output = torch.log(output.squeeze())
        loss_matrix = output.mul(q)

        return -torch.sum(loss_matrix).div(output.shape[0])

    
class RerankLoss(torch.nn.Module):
    def __init__(self, margin: float = 5e-4, reduction: str = 'mean'):
        super(RerankLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, output: torch.Tensor, labels: torch.Tensor):

        # output [B, S, 1]
        # labels [B, S]

        y_rele = labels == 1.
        y_irre = labels == 0.

        total_rele = y_rele.sum().item()
        total_irre = y_irre.sum().item()

        if total_rele == 0 or total_irre == 0:
            return t.tensor(0, requires_grad=True)

        y_pos_mean = y_rele.mul(output.squeeze()).sum().div(total_rele)
        y_neg_mean = y_irre.mul(output.squeeze()).sum().div(total_irre)

        return max(torch.tensor(0., requires_grad=True), y_neg_mean - y_pos_mean + self.margin)

        
class MtCutLoss(torch.nn.Module):
    def __init__(self, rerank_weight: float=None, classi_weight: float=None, num_tasks: float= None, tau: float=None):
        super(MtCutLoss, self).__init__()
        self.rerank_weight, self.classi_weight = rerank_weight, classi_weight # 0.5, 0.5
        self.cutloss = AttnCutLoss(tau=tau)
        self.rerankloss = RerankLoss()
        self.classiloss = torch.nn.BCELoss()
        self.num_tasks = num_tasks
        
    def forward(self, output: torch.Tensor, data):
        # output [[B, S, 1], [B, S, 1], [B, S, 1]]
        # labels [B, S]

        if self.num_tasks == 3:
            # pred_y [B, S, 1]
            # rerank_y [B, S, 1]
            # cut_y [B, S, 1]
            pred_y, rerank_y, cut_y = output

        elif self.num_tasks == 2.1:
            pred_y, cut_y = output
        else:
            rerank_y, cut_y = output

        class_label = rerank_label = data["rel"]

        cutloss = self.cutloss(cut_y, data)

        if self.num_tasks == 3 or self.num_tasks == 2.2:
            # rerank_y [B, S, 1]
            # rerank_label [B, S]
            rerankloss = self.rerankloss(rerank_y, rerank_label).mul(self.rerank_weight)

        if self.num_tasks == 3 or self.num_tasks == 2.1:
            # pred_y.squeeze() [B, S] sigmoid
            # class_label [B, S]
            classiloss = self.classiloss(pred_y.squeeze(), class_label).mul(self.classi_weight)

        if self.num_tasks == 3:
            return cutloss.add(rerankloss).add(classiloss)
        elif self.num_tasks == 2.1:
            return cutloss.add(classiloss)
        else:
            return cutloss.add(rerankloss)

