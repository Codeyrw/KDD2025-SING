import argparse
import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_scatter import scatter
from itertools import chain
#from GOOD.data import load_dataset, create_dataloader 
from logger import Logger, SimpleLogger
from dataset import load_nc_dataset
from data_utils import normalize, gen_normalized_adjs, evaluate, evaluate_whole_graph, eval_acc, eval_rocauc, eval_f1, to_sparse_tensor, load_fixed_splits
from parse import parse_method_ours,  parser_add_main_args
from model_ours import Ours, Graph_Editer

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8' 
    #torch.use_deterministic_algorithms(True)

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)
fix_seed(2)

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
def calc_loss(x, x_aug,temperature=0.2):
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)
    
    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / temperature)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    
    
    loss_0 = pos_sim / (sim_matrix.sum(dim=0) )
    loss_1 = pos_sim / (sim_matrix.sum(dim=1) )

    loss_0 = - torch.log(loss_0).mean()
    loss_1 = - torch.log(loss_1).mean()
    loss_re = (loss_0 + loss_1) / 2.0
    return loss_re

def get_dataset(dataset, ratio=None, sub_dataset=None):
    ### Load and preprocess data ###
    if dataset == 'elliptic':
        dataset = load_nc_dataset('elliptic', sub_dataset)
    else:
        raise ValueError('Invalid dataname')

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    dataset.n = dataset.graph['num_nodes']
    dataset.c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    dataset.d = dataset.graph['node_feat'].shape[1]

    dataset.graph['edge_index'], dataset.graph['node_feat'] = \
        dataset.graph['edge_index'], dataset.graph['node_feat']

    return dataset


if args.dataset == 'elliptic':
    tr_subs, val_subs, te_subs = [i for i in range(6, 11)], [i for i in range(11, 16)], [i for i in range(16, 49)]
    datasets_tr = [get_dataset(dataset='elliptic', sub_dataset=tr_subs[i]) for i in range(len(tr_subs))]
    datasets_val = [get_dataset(dataset='elliptic', sub_dataset=val_subs[i]) for i in range(len(val_subs))]
    datasets_te = [get_dataset(dataset='elliptic', sub_dataset=te_subs[i]) for i in range(len(te_subs))]
else:
    raise ValueError('Invalid dataname')
    
for i in range(len(datasets_tr)):
    dataset_tr = datasets_tr[i]
    print(f"Train num nodes {dataset_tr.n} | num classes {dataset_tr.c} | num node feats {dataset_tr.d}")
dataset_val = datasets_val[0]
print(f"Val num nodes {dataset_val.n} | num classes {dataset_val.c} | num node feats {dataset_val.d}")
for i in range(len(te_subs)):
    dataset_te = datasets_te[i]
    print(f"Test {i} num nodes {dataset_te.n} | num classes {dataset_te.c} | num node feats {dataset_te.d}")


model = Ours(args,  dataset_tr.c, dataset_tr.d, args.gnn, device)
model.train()
print('MODEL:', model)
generators = []
for dataset in datasets_tr:
    edges = dataset.graph['edge_index']
    num_edges = edges.size()[1]
    generators.append(Graph_Editer(args.K, num_edges, args.device))

criterion = nn.BCEWithLogitsLoss()
if args.rocauc or args.dataset in ('twitch-e', 'fb100', 'elliptic'):
    criterion = nn.BCEWithLogitsLoss()
    eval_func = eval_f1
else:
    criterion = nn.NLLLoss()
    eval_func = eval_acc

logger = Logger(args.runs, args)


print('DATASET:', args.dataset)

for run in range(0,args.runs):
    fix_seed(run)
    ### Load method ###
    model.reset_parameters()
    
    generators_params = []
    for generator in generators:
        generator.reset_parameters()
        generators_params.append(generator.parameters())

    #initialize optimizer
    optimizer_model = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_generator = torch.optim.AdamW(chain.from_iterable(generators_params),
                                            lr=args.lr_a, weight_decay=args.weight_decay)

    best_val = float('-inf')

    val_history = []
    for epoch in range(60):
        model.train()

        for dataset_ind in range(len(datasets_tr)):
            dataset_tr = datasets_tr[dataset_ind]
            generators_per_dataset = generators[dataset_ind]
            edge_index= dataset_tr.graph['edge_index']
            Y = torch.zeros(dataset_tr.n,dataset_tr.label.max().item()+1).to(device)#N*C
            idx=torch.arange(dataset_tr.n).to(device)
            Y[idx,dataset_tr.label.squeeze(1)]=1

            idt = torch.matmul(Y,Y.T) #train_num*train_num
            i=0
            label_sign = []
            init_label=0
            while i < dataset_tr.label.squeeze(1).shape[0]:
                if dataset_tr.label.squeeze(1)[i] == init_label:
                    label_sign.append(i)
                    i=0
                    init_label+=1
                else:
                    i+=1
                if init_label == dataset_tr.c:
                    break
            label_sign = torch.tensor(np.array(label_sign))
            #---------------#
            for inner_steps in range(args.inner_steps):
                z_all=[]
                rep_all=[]
                loss_array = []
                loss_var_array = []
                kld_array = []
                loss_og, rep,z = model(dataset_tr, criterion)
                loss_array.append(loss_og.view(-1))
                z_all.append(z)
                A =torch.sparse_coo_tensor(edge_index.cuda(), torch.ones(edge_index.shape[1]).cuda(),[dataset_tr.n,dataset_tr.n]).to(device)
                D=torch.sum(A.to_dense(),dim=1).unsqueeze(1)
                z_res = z-torch.mm(A.to_dense(),z)/(D+1e-10)
                rep_all.append(z_res)# N*d
                for k in range(0, args.K):
                    mask_per_view,_ = generators_per_dataset(k)
                    loss_local,rep,z = model(dataset_tr, criterion, mask_per_view)
                    z_all.append(z)
                    loss_array.append(loss_local.view(-1))
                    loss_var_array.append(torch.sqrt(loss_local).view(-1))
                    Ak =torch.sparse_coo_tensor(edge_index.cuda(), mask_per_view.cuda(),[dataset_tr.n,dataset_tr.n]).to(device)
                    Dk=torch.sum(Ak.to_dense(),dim=1).unsqueeze(1)
                    rep_all.append(z-torch.mm(Ak.to_dense(),z)/(Dk+1e-10))
                Sc = torch.stack(rep_all,0)#K*N*d
                loss_str = torch.norm((torch.var(Sc,dim=0)).mean(0))/args.hidden_channels
                con_loss = 0
                for k in range(0,args.K):
                    if k!= (args.K-1):
                        con_loss += calc_loss(z_all[k],z_all[k+1])
                    else:
                        con_loss += calc_loss(z_all[k],z_all[0])
                con_loss = (con_loss/args.K).to(device)
                Loss = torch.cat(loss_array, dim=0)
                Loss_var = torch.cat(loss_var_array, dim=0)
                _, Mean = torch.var_mean(Loss)
                Var, _ = torch.var_mean(Loss_var)

                
                loss_generator = Mean + 0.5*con_loss  - args.dist_weight * Var + 1*loss_str

                optimizer_generator.zero_grad()
                loss_generator.backward()
                optimizer_generator.step()

            loss_array = []
            z_c_all=[]
            loss_str_al = 0
            loss_og,rep,z = model(dataset_tr, criterion)
            loss_array.append(loss_og.view(-1))
            A =torch.sparse_coo_tensor(edge_index.cuda(), torch.ones(edge_index.shape[1]).cuda(),[dataset_tr.n,dataset_tr.n]).to(device)
            D=torch.sum(A.to_dense(),dim=1).unsqueeze(1)
            z_res = rep-torch.mm(A.to_dense(),rep)/(D+1e-10)
            ##----------------structure align in graph------------------------------------------------#
            for kk in range(z_res.shape[0]):
                z_c_all.append(torch.mean(z_res*idt[kk,:].unsqueeze(1),dim=0).unsqueeze(0))
            z_c =torch.cat(z_c_all,dim=0)

            anchor = z_c[label_sign.to(device)]#C*d

            BY_pos = torch.matmul(z_res/(torch.norm(z_res,dim=1).unsqueeze(1)+1e-10),anchor.T/(torch.norm(anchor,dim=1)+1e-10))[range(z_c.shape[0]),dataset_tr.label.squeeze(1)]
            BY_gui = torch.matmul(z_res/(torch.norm(z_res,dim=1).unsqueeze(1)+1e-10),anchor.T/(torch.norm(anchor,dim=1)+1e-10))#N*C
            mask = torch.ones(z_res.shape[0],dataset_tr.c).to(device)
            mask[range(z_c.shape[0]),dataset_tr.label.squeeze(1)]=0
            BY_neg = torch.mean(BY_gui*mask,dim=1)
            loss_str_al += torch.mean(torch.ones(BY_pos.shape[0]).to(device) - BY_pos+ 0.2*BY_neg)#

            #------------------------------------------------------------------------------------------#
            for k in range(0, args.K):
                z_c_all=[]
                mask_per_view,_ = generators_per_dataset(k)
                loss_local,rep,z = model(dataset_tr, criterion, mask_per_view)
                loss_array.append(loss_local.view(-1))
                Ak =torch.sparse_coo_tensor(edge_index.cuda(), mask_per_view.cuda(),[dataset_tr.n,dataset_tr.n]).to(device)
                Dk=torch.sum(Ak.to_dense(),dim=1).unsqueeze(1)
                z_res = rep-torch.mm(Ak.to_dense(),rep)/(Dk+1e-10)
                ##----------------structure align in graph------------------------------------------------#
                for kk in range(z_res.shape[0]):
                    z_c_all.append(torch.mean(z_res*idt[kk,:].unsqueeze(1),dim=0).unsqueeze(0))
                z_c =torch.cat(z_c_all,dim=0)
                anchor = z_c[label_sign.to(device)]#C*d

                BY_pos = torch.matmul(z_res/(torch.norm(z_res,dim=1).unsqueeze(1)+1e-10),anchor.T/(torch.norm(anchor,dim=1)+1e-10))[range(z_c.shape[0]),dataset_tr.label.squeeze(1)]
                BY_gui = torch.matmul(z_res/(torch.norm(z_res,dim=1).unsqueeze(1)+1e-10),anchor.T/(torch.norm(anchor,dim=1)+1e-10))#N*C
                mask = torch.ones(z_res.shape[0],dataset_tr.c).to(device)
                mask[range(z_c.shape[0]),dataset_tr.label.squeeze(1)]=0
                BY_neg = torch.mean(BY_gui*mask,dim=1)
                loss_str_al += torch.mean(torch.ones(BY_pos.shape[0]).to(device) - BY_pos + 0.2*BY_neg)#

            Loss = torch.cat(loss_array, dim=0)
            Var, Mean = torch.var_mean(Loss)
            optimizer_model.zero_grad()
            loss_classifier = Mean + Var + 0.2*loss_str_al/(args.K+1)
            loss_classifier.backward()
            optimizer_model.step()

        accs, test_outs = evaluate_whole_graph(args, model, datasets_tr, datasets_val, datasets_te, eval_func)
        logger.add_result(run, accs)

        if epoch % args.display_step == 0:

            print(f'Epoch: {epoch:02d}, '
                    f'Mean Loss: {Mean:.4f}, '
                    f'Var Loss: {Var:.4f}, '
                    f'loss str:{loss_str_al:.3f},'
                    f'Train: {100 * accs[0]:.2f}%, '
                    f'Valid: {100 * accs[1]:.2f}%, ')
            test_info = ''
            for test_acc in accs[2:]:
                test_info += f'Test: {100 * test_acc:.2f}% '
            print(test_info)

    logger.print_statistics(run)


### Save results ###
results = logger.print_statistics()
filename = f'./results/{args.dataset}.csv'
print(f"Saving results to {filename}")

