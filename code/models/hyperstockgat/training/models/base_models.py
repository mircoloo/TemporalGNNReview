"""Base model class."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hyperstockgat.training.layers.layers import FermiDiracDecoder
import models.hyperstockgat.training.layers.hyp_layers as hyp_layers
import models.hyperstockgat.training.manifolds as manifolds
import models.hyperstockgat.training.models.encoders as encoders
from models.hyperstockgat.training.models.decoders import model2decoder
from models.hyperstockgat.training.utils.eval_utils import acc_f1
device = 'cuda'

class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.])) # the curvature
        self.manifold = getattr(manifolds, self.manifold_name)() # get the manifold 
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1 # add one dimension for hyperboloid
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, :, :, 0:1], x], dim=3)
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)  ##EDIT HERE make it div by trch.sum(mask)


def trr_loss_mse_rank(pred, base_price, ground_truth, mask, alpha, no_stocks):
    return_ratio = torch.div((pred- base_price), base_price)
    reg_loss = weighted_mse_loss(return_ratio, ground_truth, mask)
    all_ones = torch.ones(no_stocks,1).to(device)
    pre_pw_dif =  (torch.matmul(return_ratio, torch.transpose(all_ones, 0, 1)) 
                    - torch.matmul(all_ones, torch.transpose(return_ratio, 0, 1)))
    gt_pw_dif = (
            torch.matmul(all_ones, torch.transpose(ground_truth,0,1)) -
            torch.matmul(ground_truth, torch.transpose(all_ones, 0,1))
        )

    mask_pw = torch.matmul(mask, torch.transpose(mask, 0,1))
    rank_loss = torch.mean(
            F.relu(
                ((pre_pw_dif*gt_pw_dif)*mask_pw)))
    loss = reg_loss + alpha*rank_loss
    del mask_pw, gt_pw_dif, pre_pw_dif, all_ones
    return loss, reg_loss, rank_loss, return_ratio


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
       
        self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, adj):
        output = self.decoder.decode(h, adj)
        return F.leaky_relu(output, 0.2)

    def compute_metrics(self, embeddings,adj,base_price, ground_truth, mask, alpha, no_stocks):
        output = self.decode(embeddings, adj)
        loss, reg_loss, rank_loss, return_ratio = trr_loss_mse_rank(output.reshape((1026,1)),base_price, ground_truth, mask, alpha, no_stocks)
        # acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'reg_loss': reg_loss, 'rank_loss': rank_loss, 'rr':return_ratio}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]



