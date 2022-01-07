import torch
from torchvision import datasets, transforms

import argparse
import os
from tqdm import tqdm

from model import Model
from preprocess import load_data

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


def main():
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument('--p', type=float, default=0.75, help='graph probability, (default: 0.75)')
    parser.add_argument('--c', type=int, default=78,
                        help='channel count for each node, (example: 78, 157, 267), (default: 78)')
    parser.add_argument('--k', type=int, default=4,
                        help='each node is connected to k nearest neighbors in ring topology, (default: 4)')
    parser.add_argument('--m', type=int, default=5,
                        help='number of edges to attach from a new node to existing nodes, (default: 5)')
    parser.add_argument('--graph-mode', type=str, default="NCNN",
                        help="random graph, (Example: NCNN, ER, WS, BA), (default: NCNN)")
    parser.add_argument('--graph-type', type=str, default="ncnn16.xlsx",
                        help="adjacency matrix end with '.xlsx', (Example: ncnn16.xlsx, ncnn18.xlsx), (default: ncnn.xlsx)")
    parser.add_argument('--node-num', type=int, default=16, help="Number of graph node (default n=16)")
    parser.add_argument('--batch-size', type=int, default=128, help='batch size, (default: 128)')
    parser.add_argument('--dataset-mode', type=str, default="CIFAR10",
                        help='Which dataset to use? (Example, CIFAR10, CIFAR100, MNIST), (default: CIFAR10)')
    parser.add_argument('--is-train', type=bool, default=True, help="True if training, False if test. (default: True)")

    args = parser.parse_args()

    _, test_loader = load_data(args)

    if os.path.exists("./checkpoint"):
        model = Model(args.node_num, args.p, args.c, args.c, args.graph_mode, args.model_mode, args.dataset_mode, args.is_train).to(device)
        filename = "graph_mode_" + args.graph_mode + "_graph_type_" + args.graph_type + "_node_num_" + str(args.node_num)  + "_dataset_" + args.dataset_mode
        checkpoint = torch.load('./checkpoint/' + filename + 'ckpt.t7')
        model.load_state_dict(checkpoint['model'])
        end_epoch = checkpoint['epoch']
        best_acc = checkpoint['acc']
        print("[Saved Best Accuracy]: ", best_acc, '%', "[End epochs]: ", end_epoch)

        model.eval()
        correct = 0
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            y_pred = output.data.max(1)[1]
            correct += y_pred.eq(target.data).sum()
        print("[Test Accuracy] ", 100. * float(correct) / len(test_loader.dataset), '%')

    else:
        assert os.path.exists("./checkpoint/" + str(args.seed) + "ckpt.t7"), "File not found. Please check again."
    print("Number of model parameters: ", get_model_parameters(model))


if __name__ == "__main__":
    main()
