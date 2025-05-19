This repository is the implementation of [paper](https://www.sciencedirect.com/science/article/pii/S0957417424008297) published in Expert Systems with Applications

## Installation

Install [pytorch](https://pytorch.org/get-started/locally/)

Install [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

## Data Preprocessing
We use data provided by GTN(https://github.com/seongjunyun/Graph_Transformer_Networks).

Take DBLP as an example to show the formats of input data:

`node_features.pkl` is a numpy array whose shape is (num_of_nodes, num_of_features). It contains input node features. 

`edges.pkl` is a list of scipy sparse matrices. Each matrix has a shape of (num_of_nodes, num_of_nodes) and is formed by edges of a certain edge type.

`labels.pkl` is a list of lists. labels[0] is a list containing training labels and each item in it has the form [node_id, target]. labels[1] and labels[2] are validation labels and test labels respectively with the same format.

## Running the code
``` 
$ mkdir data
$ cd data
```
Download datasets (DBLP, ACM, IMDB) and extract file into data folder.
```
$ cd ..
```
- ACM
- NodeAttention Dropout=0.3  lr=0.005   head=6
- RelationAttention Dropout=0.5  lr=0.005
- class=3
- L2—norm=False
```
python ACM_run.py --norm false
```
- IMDB
- NodeAttention Dropout=0.5  lr=0.02   head=8
- RelationAttention Dropout=0.5  lr=0.02
- class=3
- L2—norm=True
```
python IMDB_run.py --norm true
```
- DBLP    
- NodeAttention Dropout=0.2  lr=0.01   head=8
- RelationAttention Dropout=0.2  lr=0.01
- class=4 
- L2—norm=True
```
python DBLP_run.py --norm true
```

