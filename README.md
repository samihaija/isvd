# isvd
Official implementation of NeurIPS'21: Implicit SVD for Graph Representation Learning

If you find this code useful, you may cite us as:
```
@inproceedings{haija2021isvd,
  author={Sami Abu-El-Haija AND Hesham Mostafa AND Marcel Nassar AND Valentino Crespi AND Greg Ver Steeg AND Aram Galstyan},
  title={Implicit SVD for Graph Representation Learning},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021},
}
```


## To run link prediction on Stanford SNAP and node2vec datasets:

To embed with rank-32 SVD:
```
python3 run_snap_linkpred.py --dataset_name=ppi --dim=32
python3 run_snap_linkpred.py --dataset_name=ca-AstroPh --dim=32
python3 run_snap_linkpred.py --dataset_name=ca-HepTh --dim=32
python3 run_snap_linkpred.py --dataset_name=soc-facebook --dim=32
```

To embed with rank 256 on half of the training edges, determine "best rank"
based on the remaining half, then re-run sVD with the best rank on all of training:
(note: negative `dim` causes this logic):
```
python3 run_snap_linkpred.py --dataset_name=ppi --dim=-256
python3 run_snap_linkpred.py --dataset_name=ca-AstroPh --dim=-256
python3 run_snap_linkpred.py --dataset_name=ca-HepTh --dim=-256
python3 run_snap_linkpred.py --dataset_name=soc-facebook --dim=-256
```

## To run semi-supervised node classification on Planetoid datasets
You must first download the planetoid dataset as:

```
mkdir -p ~/data
cd ~/data
git clone git@github.com:kimiyoung/planetoid.git
```

Afterwards, you may navigate back to this directory and run our code as:
```
python3 run_planetoid.py --dataset=ind.citeseer
python3 run_planetoid.py --dataset=ind.cora
python3 run_planetoid.py --dataset=ind.pubmed
```

## To run link prediction on Stanford OGB DDI
```
python3 ogb_linkpred_sing_val_net.py
```
Note the above will download the dataset from Stanford. If you already have it, you may symlink it into directory `dataset`


## To run link prediction on Stanford OGB ArXiv
As our code imports gttf, you must first clone it onto the repo:
```
git clone git@github.com:isi-usc-edu/gttf.git
```

Afterwards, you may run as:
```
python3 final_obgn_mixed_device.py --funetune_device='gpu:0'
```
Note the above will download the dataset from Stanford. If you already have it, you may symlink it into directory `dataset`. You may skip the `finetune_device` argument if you do not have a GPU installed.