# A Structure-aware Invariant Learning Framework for Node-level Graph OOD Generalization

This is the pytorch implementation for "A Structure-aware Invariant Learning Framework for Node-level Graph OOD Generalization" in KDD 2025.

## Environment requirements

We apply **Python 3.7.16** to run the code and some necessary requirements are listed as follows:

python==3.7.16  
torch==1.13.1  
torch_geometric ==2.2.0  
numpy==1.21.5  
scikit_learn==1.0.2  
networkx==2.6.3  
rdkit==2023.03.2  
scipy==1.7.3  
ogb==1.3.6 
gdown==4.4.0  
munch==2.5.0  
ruamel.yaml==0.17.21  
protobuf==3.20.1  
tensorboard==2.8.0  
tqdm==4.64.0  

## Download the datasets
The transductive datasets are provided by the [GOOD-benchmark](https://github.com/divelab/GOOD/tree/main). For Inductive datasets, we use the datasets from [LisA](https://github.com/Samyu0304/LiSA). For convenience, we have uploaded the zipped dataset to the [Google Drive link](https://drive.google.com/drive/folders/1SgxiUBQV6gOs4TVL_uTGVDGBcqOhRtjQ). For transductive datasets and OGB-Arxiv, you can download and unzip them to the directory './arxiv/data'. For inductive Elliptic and Twitch-Explicit, the datasets can be downloaded to directories './elliptic/data' and './multigraph/data', respectively.  


## Instructions to run the code

Please see run.sh and follow the given commands to run the code on datasets under transductive and inductive settings.

## Reference

Welcome to kindly cite our work with:  
@inproceedings{yuan2025structure,  
  title={A Structure-aware Invariant Learning Framework for Node-level Graph OOD Generalization},  
  author={Yuan, Ruiwen and Tang, Yongqiang and Zhang, Wensheng},  
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1},  
  pages={1879--1890},  
  year={2025}  
}  
