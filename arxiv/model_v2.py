import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from nets_ours import *
#from nets_ours_v2 import *

class Ours(nn.Module):
    def __init__(self, args, c, d, gnn, device, dataset=None):
        super(Ours, self).__init__()
        if gnn == 'gcn':
            self.gnn = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=not args.no_bn).to(device)
        elif gnn == 'gcnii':
            self.gnn = GCN2Net(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        lam = args.gcnii_lamda,
                        alp = args.gcnii_alpha).to(device)
        elif args.gnn == 'sggcn':
            self.gnn = SGGCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=not args.no_bn).to(device)
        elif args.gnn == 'appnp':
            self.gnn = APPNPNet(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        dropout=args.dropout,
                        k = args.num_layers,
                        alpha = args.gpr_alpha).to(device)

        elif gnn == 'sage':
            self.gnn = OurSAGE(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout).to(device)

        self.device = device
        self.args = args

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def energy_scores(self, logits):
        exp_logits = torch.exp(logits)
        sum_exp_logits = torch.sum(exp_logits, dim=-1)

        return torch.log(sum_exp_logits)


    def forward(self, data, criterion, mask=None,node_mask=None):

        x, y = data.graph['node_feat'].to(self.device), data.label.to(self.device)
        edge_index = data.graph['edge_index'].to(self.device)
        if node_mask != None:
            out,rep,z = self.gnn(x+node_mask, edge_index, mask)#
        else:
            out,rep,z = self.gnn(x, edge_index, mask)
        #print(out)
        #out = torch.clamp(out, -10, 10)
        if self.args.dataset == 'ogb-arxiv':
            loss = self.sup_loss(y, out, criterion)
        else:
            #print(out[data.train_mask].shape)
            loss = self.sup_loss(y[data.train_mask], out[data.train_mask], criterion) 

        #scores = self.energy_scores(out)
        return loss,rep,z

    def inference(self, data, mask=None):
        x = data.graph['node_feat'].to(self.device)
        edge_index = data.graph['edge_index'].to(self.device)
        out,_,_ = self.gnn(x, edge_index, mask)
        return out

    def sup_loss(self, y, pred, criterion):
        if self.args.dataset in ('twitch-e', 'fb1001', 'fb1002','fb1003','elliptic'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            loss = criterion(pred, true_label.squeeze(1).to(torch.float))
        else:
            criterion = nn.CrossEntropyLoss()
            #out = F.log_softmax(pred, dim=1)
            target = y.squeeze(1)
            loss = criterion(pred, target)
        return loss


class Graph_Editer(nn.Module):
    def __init__(self, K, edge_num,node_num, node_dim, device):
        super(Graph_Editer, self).__init__()
        self.K = K
        self.edge_num = edge_num
        self.S = 0.01
        self.sample_size = int(self.S * edge_num)
        self.B = nn.Parameter(torch.FloatTensor(K, edge_num))
        #self.M = nn.Parameter(torch.FloatTensor(K, node_num, node_dim))

        #print(self.B)
        self.epsilon = 0.000001
        self.temperature = 1.0

        self.device = device

    def reset_parameters(self):
        nn.init.uniform_(self.B)
        #nn.init.uniform_(self.M)
    def kld(self, mask):
        pos = mask
        neg = 1 - mask
        kld_loss = torch.mean(pos * torch.log(pos/0.5 + 0.00000001) + neg * torch.log(neg/0.5 + 0.000000001))

        return kld_loss

    def forward(self, k,edge_index,n):
        #return a KL-like loss term to control the information flow
        Bk = self.B[k]
        #Mk= self.M[k]
        #print(Mk)
        #print(Bk)
        #print(Mk)
        #mask = torch.clamp(Bk, -10,10).to(self.device)
        #data = edge_index.shape[1]
        #A = to_dense_adj(edge_index,max_num_nodes=n)[0].to(torch.int).to(self.device)
        mask = torch.sigmoid(Bk).to(self.device)
        #mask = torch.sparse_coo_tensor(edge_index.to(self.device), Bk_tan, A.shape).to_dense().to(self.device)
        #print(mask)
        #A_c = torch.ones(n, n, dtype=torch.int).to(self.device) - A
        #M= A+ Bk_tan*(A_c-A)
        #sample_mask =  self.straight_through(mask) 
        #node_mask = torch.clamp(Mk, -10,10).to(self.device)
        #node_mask = torch.sigmoid(Mk).to(self.device)
        #print(torch.isnan(node_mask).any())
        #print(node_mask)
        #mask = mask.unsqueeze(dim = 0)
        #cat_mask = torch.cat([mask,1-mask],dim = 0)
        #sample_mask = self.straight_through(mask)#F.gumbel_softmax(cat_mask, tau=1, dim = 0, hard=False)
        #kld_loss = self.kld(mask)
        #print(sample_mask[0])
        #print(kld_loss)
        Bk = self.B[k]
        '''Bk = self.B[k]
        mask = torch.clamp(Bk, -10, 10).to(self.device)
        mask = torch.sigmoid(mask)
        #mask = mask.unsqueeze(dim = 0)
        #cat_mask = torch.cat([mask,1-mask],dim = 0)
        sample_mask = self.straight_through(mask)#F.gumbel_softmax(cat_mask, tau=1, dim = 0, hard=False)'''
        #kld_loss = self.kld(mask)
        #print(sample_mask[0])
        return mask#,node_mask #,kld_loss#, node_mask

    def sample(self, k):
        Bk = self.B[k]
        mask = torch.clamp(Bk, -10, 10).to(self.device)
        mask = torch.sigmoid(mask)
        mask = self.straight_through(mask)#torch.sigmoid(mask)

        #mask = mask.unsqueeze(dim=0)
        #cat_mask = torch.cat([mask, 1 - mask], dim=0)
        #sample_mask = F.gumbel_softmax(cat_mask, tau=1, dim=0, hard=False)
        #print(sample_mask)

        return mask
    #straight-through sampling

    def straight_through(self, mask):
        _, idx = torch.topk(mask, self.sample_size)
        sample_mask = torch.zeros_like(mask).to(self.device)
        sample_mask[idx]=1

        return sample_mask + mask - mask.detach()