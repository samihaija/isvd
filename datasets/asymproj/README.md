# AsymProj Datasets

These datasets are used in WatchYourStep paper. We copy them here to maintain
the train/test splits, but the original sources are:

 * `ppi` (Protein-Protein Interactions) came from node2vec paper (which in turn, processed it from BioGrid)
 * `ego-facebook`, `ca-HepTh`, and `ca-AstroPh` came from stanford's snap

Each directory contains 3 files:

 1. `train.txt.npy`: numpy int array of shape (trainEdges, 2)
 1. `test.txt.npy`: numpy int array of shape (testEdges, 2)
 1. `test.neg.txt.npy`: numpy int array of shape (testNegEdges, 2)


The union of the first two are the input graph. trainEdges = testEdges = testNegEdges (per WatchYourStep & AsymProj, mimicing node2vec). The last np file contains negative edges sampled uniformly at random from the graph compliment.

## References:
 * node2vec: Grover & Leskovec, *node2vec: Scalable Feature Learning for Networks*, KDD 2016
 * AsymProj: Abu-El-Haija et al, *Learning Edge Representations via Low-Rank Asymmetric Projections*, CIKM 2017.
 * WatchYourStep (WYS): Abu-El-Haija et al, *Watch Your Step: Learning Node Embeddings via Graph Attention*, NeurIPS 2018
