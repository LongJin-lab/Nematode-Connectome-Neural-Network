# Nematode Connectome Neural Network
- Implement Exploring Nematode Connectome Neural Network for Image Recognition :)

## Run
```
python main.py
```
- If you want to change hyper-parameters, you can check "python main.py --help"

Options:
- `--epochs` (int) - number of epochs, (default: 100).
- `--p` (float) - graph probability, (default: 0.75).
- `--c` (int) - channel count for each node, (example: 78, 157, 267), (default: 78).
- `--k` (int) - each node is connected to k nearest neighbors in ring topology, (default: 4).
- `--m` (int) - number of edges to attach from a new node to existing nodes, (default: 5).
- `--graph-mode` (str) - kinds of random graph, (exampple: NCNN, ER, WS, BA), (default: NCNN).
- `--node-num` (int) - number of graph node (default n=16).
- `--learning-rate` (float) - learning rate, (default: 1e-1).
- `--graph-type` (str) -adjacency matrix end with '.xlsx', (Example: ncnn16.xlsx, ncnn18.xlsx), (default: ncnn.xlsx), 
- `--batch-size` (int) - batch size, (default: 128).
- `--dataset-mode` (str) - which dataset you use, (example: CIFAR10, CIFAR100, MNIST), (default: CIFAR10).
- `--is-train` (bool) - True if training, False if test. (default: True).
- `--load-model` (bool) - (default: False).

## Test
```
python test.py
```
- Put the saved model file in the checkpoint folder and saved graph file in the saved_graph folder and type "python test.py".
- If you want to change hyper-parameters, you can check "python test.py --help"
- The model file currently in the checkpoint folder is a model with an accuracy of 92.70%.

Options:

- `--p` (float) - graph probability, (default: 0.75).
- `--c` (int) - channel count for each node, (example: 78, 157, 267), (default: 78).
- `--k` (int) - each node is connected to k nearest neighbors in ring topology, (default: 4).
- `--m` (int) - number of edges to attach from a new node to existing nodes, (default: 5).
- `--graph-mode` (str) - kinds of random graph, (exampple: NCNN, ER, WS, BA), (default: NCNN).
- `--node-num` (int) - number of graph node (default n=16).
- `--graph-type` (str) -adjacency matrix end with '.xlsx', (Example: ncnn16.xlsx, ncnn18.xlsx), (default: ncnn.xlsx), 
- `--batch-size` (int) - batch size, (default: 128).
- `--dataset-mode` (str) - which dataset you use, (example: CIFAR10, CIFAR100, MNIST), (default: CIFAR10).
- `--is-train` (bool) - True if training, False if test. (default: True).

## Version
- Python 3.7
- Cuda 10.2.89
- Cudnn 7.6.5
- pytorch 1.4.0
- networkx 2.3

  
