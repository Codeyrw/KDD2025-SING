#!/usr/bin/env bash

#Transductive Datasets
cd ./arxiv/
# CBAS
python v_main_CBAS_con.py --gnn sggcn --lr 0.05 --lr_a 0.05 --dropout 0.2 --K 2 --device 2 --dist_weight 0.005 --inner_steps 10 --dataset 'GOODCBAS'
python v_main_CBAS_cov.py --gnn gcn --lr 0.01 --lr_a 0.01 --dropout 0.2 --K 2 --device 0 --dist_weight 0.05 --inner_steps 10 --dataset 'GOODCBAS'

#WebKB
python v_main_WebKB_con.py --gnn gcn --lr 0.01 --lr_a 0.01 --dropout 0.0 --K 3 --device 3 --dist_weight 0.05 --inner_steps 10 --dataset 'GOODWebKB'
python v_main_WebKB_cov.py --gnn gcn --lr 0.05 --lr_a 0.05 --dropout 0.0 --K 3 --device 1 --dist_weight 0.05 --inner_steps 10 --dataset 'GOODWebKB'

#Cora
python v_main_Cora_condeg.py --gnn sggcn --lr 0.03 --lr_a 0.03 --dropout 0.5 --K 2 --device 2 --dist_weight 0.05 --inner_steps 3 --dataset 'GOODCora'
python v_main_Cora_covdeg.py --gnn sggcn --lr 0.03 --lr_a 0.03 --dropout 0.5 --K 2 --device 2 --dist_weight 0.05 --inner_steps 10 --dataset 'GOODCora'
python v_main_Cora_conword.py --gnn sggcn --lr 0.02 --lr_a 0.02 --dropout 0.5 --K 2 --device 2 --dist_weight 0.05 --inner_steps 10 --dataset 'GOODCora'
python v_main_Cora_covword.py --gnn sggcn --lr 0.02 --lr_a 0.02 --dropout 0.5 --K 2 --device 2 --dist_weight 0.05 --inner_steps 10 --dataset 'GOODCora'

#Arxiv
python v_main_Arxiv_condeg.py --gnn sggcn --lr 0.001 --lr_a 0.001 --dropout 0.5 --K 2 --device 3 --dist_weight 0.05 --inner_steps 1 --dataset 'GOODArxiv'
python v_main_Arxiv_covdeg.py --gnn sggcn --lr 0.0005 --lr_a 0.0005 --dropout 0.5 --K 2 --device 1 --dist_weight 0.05 --inner_steps 1 --dataset 'GOODArxiv'
python v_main_Arxiv_contim.py --gnn sggcn --lr 0.001 --lr_a 0.001 --dropout 0.5 --K 1 --device 0 --dist_weight 0.05 --inner_steps 1 --dataset 'GOODArxiv'
python v_main_Arxiv_covtim.py --gnn sggcn --lr 0.005 --lr_a 0.001 --dropout 0.5 --K 2 --device 2 --dist_weight 0.05 --inner_steps 1 --dataset 'GOODArxiv'

#Inductive Datasets

#OGB-Arxiv
cd ./arxiv/
python v_main_ogbarxiv.py --gnn sggcn --lr 0.01 --lr_a 0.005 --dropout 0.3 --K 2 --device 2 --dist_weight 0.005 --inner_steps 2

#Ellpitic
cd ./elliptic/
python v_main_elliptic.py --dist_weight 0.05  --kld_weight 0.01 --epochs 100  --gnn sage --weight_decay 0.0 --lr 0.005 --dropout 0.5

#Twitch-Explicit
cd ./multigraph/
python v_main_twitch.py --dataset twitch-e --gnn gcn --lr 0.03 --display_step 99 --warm_up 20 --early_stopping 10 --weight_decay 1e-3 --K 2 --num_layers 2 --device 3 --kld_weight 0.0 --runs 10 --rocauc

